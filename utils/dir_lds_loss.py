"""
DIR-style LDS Loss (Label Distribution Smoothing)
Based on: Delving into Deep Imbalanced Regression (ICML 2021)
Repository: https://github.com/YyzHarry/imbalanced-regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import convolve1d


def get_lds_kernel_window(kernel, ks, sigma):
    """
    Compute LDS kernel weights for smoothing.
    
    Args:
        kernel: 'gaussian', 'triang', or 'laplace'
        ks: kernel size (must be odd)
        sigma: bandwidth parameter
    """
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:  # laplace
        kernel_window = laplace(ks, sigma)
    
    return kernel_window


def gaussian_filter1d(input_array, sigma):
    """1D Gaussian filter (scipy-compatible)"""
    from scipy.ndimage import gaussian_filter1d as scipy_gaussian
    return scipy_gaussian(input_array, sigma=sigma)


def triang(ks):
    """Triangular kernel"""
    return np.array([1 - abs((i - (ks - 1) / 2) / ((ks + 1) / 2)) for i in range(ks)])


def laplace(ks, sigma):
    """Laplace kernel"""
    half_ks = (ks - 1) // 2
    return np.array([np.exp(-abs(i - half_ks) / sigma) for i in range(ks)])


class DIR_LDS_Loss(nn.Module):
    """
    DIR-style LDS reweighted regression loss.
    
    Performs Label Distribution Smoothing and inverse density reweighting
    exactly as in the original DIR paper.
    
    Args:
        train_targets: numpy array of training labels (1D)
        base: base loss type ('l1', 'smoothl1', or 'mse')
        smoothl1_beta: beta parameter for SmoothL1Loss (default: 0.1)
        reweight: reweighting scheme ('inv' or 'sqrt_inv')
        lds: whether to apply LDS smoothing
        lds_kernel: kernel type for LDS ('gaussian', 'triang', 'laplace')
        lds_ks: kernel size (must be odd)
        lds_sigma: kernel bandwidth
        num_bins: number of bins for discretization (default: 1000)
    """
    
    def __init__(
        self,
        train_targets,
        base='smoothl1',
        smoothl1_beta=0.1,
        reweight='inv',
        lds=True,
        lds_kernel='gaussian',
        lds_ks=5,
        lds_sigma=2.0,
        num_bins=1000
    ):
        super().__init__()
        
        assert base in ['l1', 'smoothl1', 'mse'], f"Unsupported base loss: {base}"
        assert reweight in ['inv', 'sqrt_inv'], f"Unsupported reweight: {reweight}"
        
        self.base = base
        self.smoothl1_beta = smoothl1_beta
        self.reweight = reweight
        self.lds = lds
        self.num_bins = num_bins
        
        # Create bins
        y_min, y_max = train_targets.min(), train_targets.max()
        bin_edges = np.linspace(y_min, y_max, num_bins + 1)
        self.register_buffer('bin_edges', torch.from_numpy(bin_edges).float())
        
        # Compute histogram (label density)
        hist, _ = np.histogram(train_targets, bins=bin_edges)
        
        # Apply LDS smoothing if enabled
        if lds:
            kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            eff_label_dist = convolve1d(hist, weights=kernel_window, mode='constant')
            print(f"✅ LDS smoothing applied: kernel={lds_kernel}, ks={lds_ks}, sigma={lds_sigma}")
        else:
            eff_label_dist = hist
            print(f"ℹ️  LDS disabled - using raw label distribution")
        
        # Compute reweighting (inverse density)
        eff_label_dist = eff_label_dist + 1e-6  # avoid division by zero
        
        if reweight == 'inv':
            weights = 1.0 / eff_label_dist
        else:  # sqrt_inv
            weights = 1.0 / np.sqrt(eff_label_dist)
        
        # Normalize weights to mean=1.0 (keeps scale consistent)
        weights = weights / weights.mean()
        
        self.register_buffer('weights', torch.from_numpy(weights).float())
        
        print(f"📊 DIR-LDS Loss initialized:")
        print(f"   Base loss: {base}" + (f" (beta={smoothl1_beta})" if base == 'smoothl1' else ""))
        print(f"   Reweight: {reweight}")
        print(f"   Bins: {num_bins}")
        print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   Weight mean: {weights.mean():.3f} (should be ≈1.0)")
    
    def forward(self, pred, target):
        """
        Compute weighted loss.
        
        Args:
            pred: predictions (B, 1) or (B,)
            target: ground truth (B, 1) or (B,)
        
        Returns:
            scalar loss
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Find bin indices for each target using bucketize (more robust than searchsorted)
        bin_indices = torch.bucketize(target, self.bin_edges) - 1
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        
        # Get weights for each sample
        sample_weights = self.weights[bin_indices]
        
        # Compute base loss
        if self.base == 'l1':
            loss = F.l1_loss(pred, target, reduction='none')
        elif self.base == 'smoothl1':
            loss = F.smooth_l1_loss(pred, target, reduction='none', beta=self.smoothl1_beta)
        else:  # mse
            loss = F.mse_loss(pred, target, reduction='none')
        
        # Apply reweighting
        weighted_loss = loss * sample_weights
        
        return weighted_loss.mean()
