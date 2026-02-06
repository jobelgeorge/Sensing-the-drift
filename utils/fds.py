"""
FDS: Feature Distribution Smoothing
Based on: Delving into Deep Imbalanced Regression (ICML 2021)
Repository: https://github.com/YyzHarry/imbalanced-regression

FDS calibrates feature statistics (mean, std) per label bin to reduce
feature imbalance and improve predictions on rare target values.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import convolve1d


def get_smoothing_kernel(kernel, ks, sigma):
    """
    Generate smoothing kernel for FDS.
    
    Args:
        kernel: 'gaussian', 'triang', or 'laplace'
        ks: kernel size (must be odd)
        sigma: bandwidth parameter
    """
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    
    if kernel == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        base = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base, sigma=sigma)
        kernel_window = kernel_window / kernel_window.max()
    elif kernel == 'triang':
        kernel_window = np.array([1 - abs((i - half_ks) / ((ks + 1) / 2)) for i in range(ks)])
    else:  # laplace
        kernel_window = np.array([np.exp(-abs(i - half_ks) / sigma) for i in range(ks)])
    
    return kernel_window


class FDS(nn.Module):
    """
    Feature Distribution Smoothing module.
    
    Maintains running statistics (mean, std) of features per label bin,
    smooths them with a kernel, and calibrates features during training/inference.
    
    Args:
        feat_dim: dimension of penultimate features
        num_bins: number of bins (should match LDS loss)
        kernel: smoothing kernel type
        ks: kernel size
        sigma: kernel bandwidth
        momentum: EMA momentum for running stats (0.9 = keep 90% old, 10% new)
        start_update: epoch to start collecting statistics
        start_smooth: epoch to start applying smoothing calibration
    """
    
    def __init__(
        self,
        feat_dim,
        num_bins=1000,
        kernel='gaussian',
        ks=5,
        sigma=2.0,
        momentum=0.9,
        start_update=1,
        start_smooth=1
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.num_bins = num_bins
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth
        
        # Running statistics per bin (exponential moving average)
        self.register_buffer('running_mean', torch.zeros(num_bins, feat_dim))
        self.register_buffer('running_var', torch.ones(num_bins, feat_dim))
        self.register_buffer('running_count', torch.zeros(num_bins))
        
        # Smoothed statistics (computed per epoch)
        self.register_buffer('smoothed_mean', torch.zeros(num_bins, feat_dim))
        self.register_buffer('smoothed_var', torch.ones(num_bins, feat_dim))
        
        # Smoothing kernel
        kernel_window = get_smoothing_kernel(kernel, ks, sigma)
        self.register_buffer('kernel', torch.from_numpy(kernel_window).float())
        
        print(f"📊 FDS initialized:")
        print(f"   Feature dim: {feat_dim}")
        print(f"   Bins: {num_bins}")
        print(f"   Kernel: {kernel}, ks={ks}, sigma={sigma}")
        print(f"   Momentum: {momentum}")
        print(f"   Start update: epoch {start_update}")
        print(f"   Start smooth: epoch {start_smooth}")
    
    @torch.no_grad()
    def update_last_epoch_stats(self, epoch):
        """
        Smooth running statistics at the end of an epoch.
        Called once per epoch (not per batch).
        """
        if epoch < self.start_smooth:
            return
        
        # Only smooth bins that have samples
        active_bins = self.running_count > 0
        
        if active_bins.sum() == 0:
            return
        
        # Smooth mean (per feature dimension)
        for d in range(self.feat_dim):
            smoothed = convolve1d(
                self.running_mean[:, d].cpu().numpy(),
                weights=self.kernel.cpu().numpy(),
                mode='constant'
            )
            self.smoothed_mean[:, d] = torch.from_numpy(smoothed).to(self.running_mean.device)
        
        # Smooth variance (per feature dimension)
        for d in range(self.feat_dim):
            smoothed = convolve1d(
                self.running_var[:, d].cpu().numpy(),
                weights=self.kernel.cpu().numpy(),
                mode='constant'
            )
            self.smoothed_var[:, d] = torch.from_numpy(smoothed).to(self.running_var.device)
        
        # Ensure variance > 0
        self.smoothed_var = torch.clamp(self.smoothed_var, min=1e-6)
    
    @torch.no_grad()
    def update_running_stats(self, feats, labels, bin_edges):
        """
        Update running statistics with new batch.
        Uses exponential moving average.
        
        Args:
            feats: (B, feat_dim)
            labels: (B,)
            bin_edges: (num_bins+1,) tensor from LDS loss
        """
        # Assign samples to bins using bucketize (more robust than searchsorted)
        bin_indices = torch.bucketize(labels, bin_edges) - 1
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        
        # Update stats for each bin
        for b in range(self.num_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            
            bin_feats = feats[mask]  # (N, feat_dim)
            
            # Compute batch statistics
            batch_mean = bin_feats.mean(dim=0)  # (feat_dim,)
            batch_var = bin_feats.var(dim=0, unbiased=False)  # (feat_dim,)
            
            # Exponential moving average
            if self.running_count[b] == 0:
                # First batch for this bin - initialize
                self.running_mean[b] = batch_mean
                self.running_var[b] = batch_var
                self.running_count[b] = mask.sum().float()
            else:
                # EMA update
                self.running_mean[b] = self.momentum * self.running_mean[b] + (1 - self.momentum) * batch_mean
                self.running_var[b] = self.momentum * self.running_var[b] + (1 - self.momentum) * batch_var
                self.running_count[b] += mask.sum().float()
    
    def forward(self, feats, y, bin_edges, epoch, train=True):
        """
        Apply FDS calibration to features using PER-BIN normalization.
        
        This is the critical DIR innovation: instead of normalizing by batch-wide stats
        (which mixes all bins), we normalize each sample by its bin's running stats,
        then map to that bin's smoothed stats.
        
        Args:
            feats: (B, feat_dim) penultimate features
            y: (B,) labels
            bin_edges: (num_bins+1,) tensor from LDS loss
            epoch: current epoch number
            train: whether in training mode
        
        Returns:
            calibrated features (B, feat_dim)
        """
        if train and epoch >= self.start_update:
            # Update running statistics (no grad needed)
            self.update_running_stats(feats.detach(), y.detach(), bin_edges)
        
        # If smoothing not started yet, return features as-is
        if epoch < self.start_smooth:
            return feats
        
        # Apply calibration with PER-BIN normalization
        with torch.no_grad():
            # Assign samples to bins using bucketize
            bin_indices = torch.bucketize(y, bin_edges) - 1
            bin_indices = bin_indices.clamp(0, self.num_bins - 1)
            
            # Get running stats for each sample's bin (current distribution)
            mu_run = self.running_mean[bin_indices]  # (B, feat_dim)
            var_run = self.running_var[bin_indices].clamp(min=1e-6)  # (B, feat_dim)
            std_run = torch.sqrt(var_run)
            
            # Get smoothed stats for each sample's bin (target distribution)
            mu_smooth = self.smoothed_mean[bin_indices]  # (B, feat_dim)
            var_smooth = self.smoothed_var[bin_indices].clamp(min=1e-6)  # (B, feat_dim)
            std_smooth = torch.sqrt(var_smooth)
            
            # Per-bin calibration: z-score by bin's running stats, then map to bin's smoothed stats
            # number of samples per bin
            bin_counts = self.running_count[bin_indices]

            # define many-shot threshold (tune this)
            many_shot_mask = bin_counts > 50  

            # default: no calibration
            feats_calibrated = feats.clone()

            # apply FDS only to non-many-shot samples
            if (~many_shot_mask).any():
                idx = ~many_shot_mask
                feats_z = (feats[idx] - mu_run[idx]) / std_run[idx]
                feats_calibrated[idx] = feats_z * std_smooth[idx] + mu_smooth[idx]
        
        return feats_calibrated
