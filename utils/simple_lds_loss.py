"""
Simple LDS Loss - Official DIR Implementation (Simplified)
Based on: https://github.com/YyzHarry/imbalanced-regression

This is a CLEAN, SIMPLE implementation that just does LDS reweighting.
NO complex adaptive mixing, NO inverse weights, NO constraint penalties.
Just pure LDS as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d as scipy_gaussian_filter1d


def get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2.0):
    """
    Generate LDS kernel for smoothing label distribution.
    
    Args:
        kernel: 'gaussian', 'triang', or 'laplace' (or 'laplacian')
        ks: kernel size (must be odd)
        sigma: bandwidth parameter
    
    Returns:
        kernel_window: normalized kernel weights
    """
    # Normalize kernel name
    if kernel == 'laplacian':
        kernel = 'laplace'
    assert kernel in ['gaussian', 'triang', 'laplace']
    assert ks % 2 == 1, "Kernel size must be odd"
    
    half_ks = (ks - 1) // 2
    
    if kernel == 'gaussian':
        # Create base impulse
        base_kernel = np.array([0.] * half_ks + [1.] + [0.] * half_ks, dtype=np.float32)
        # Apply Gaussian smoothing
        kernel_window = scipy_gaussian_filter1d(base_kernel, sigma=sigma)
        # Normalize
        kernel_window = kernel_window / kernel_window.sum()
    elif kernel == 'triang':
        # Triangular kernel
        kernel_window = np.array([1 - abs((i - half_ks) / ((ks + 1) / 2)) for i in range(ks)])
        kernel_window = kernel_window / kernel_window.sum()
    else:  # laplace
        # Laplace kernel
        kernel_window = np.array([np.exp(-abs(i - half_ks) / sigma) for i in range(ks)])
        kernel_window = kernel_window / kernel_window.sum()
    
    return kernel_window


class SimpleLDSLoss(nn.Module):
    """
    Simple Label Distribution Smoothing Loss.
    
    This is the OFFICIAL implementation from the DIR paper, simplified.
    
    Key steps:
    1. Bin training labels into fixed bins
    2. Count samples per bin (empirical distribution)
    3. Smooth the distribution with a kernel (effective distribution)
    4. Compute inverse weights: weight = 1 / effective_count
    5. Apply weights during loss computation
    
    Args:
        train_targets: numpy array of training labels
        base: base loss type ('l1', 'smoothl1', 'mse')
        smoothl1_beta: beta for smooth L1 loss
        reweight: 'inv' (1/count) or 'sqrt_inv' (1/sqrt(count))
        lds: whether to enable LDS smoothing
        lds_kernel: kernel type for smoothing
        lds_ks: kernel size
        lds_sigma: kernel bandwidth
        num_bins: number of bins to discretize labels
    """
    
    def __init__(
        self,
        train_targets,
        base='smoothl1',
        smoothl1_beta=0.1,
        reweight='sqrt_inv',
        lds=True,
        lds_kernel='gaussian',
        lds_ks=5,
        lds_sigma=2.0,
        num_bins=100
    ):
        super().__init__()
        
        assert base in ['l1', 'smoothl1', 'mse']
        assert reweight in ['inv', 'sqrt_inv']
        
        self.base = base
        self.smoothl1_beta = smoothl1_beta
        self.reweight = reweight
        self.lds = lds
        self.num_bins = num_bins
        
        # Convert to numpy if needed
        if isinstance(train_targets, torch.Tensor):
            train_targets = train_targets.cpu().numpy()
        train_targets = train_targets.flatten()
        
        # Create bin edges
        y_min, y_max = float(train_targets.min()), float(train_targets.max())
        bin_edges = np.linspace(y_min, y_max, num_bins + 1)
        self.register_buffer('bin_edges', torch.from_numpy(bin_edges).float())
        
        # Count samples per bin (empirical distribution)
        bin_counts, _ = np.histogram(train_targets, bins=bin_edges)
        
        # Apply LDS smoothing if enabled
        if lds:
            kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            # Smooth the distribution
            eff_label_dist = convolve1d(bin_counts.astype(float), weights=kernel_window, mode='constant')
            print(f"✅ LDS enabled: {lds_kernel} kernel (ks={lds_ks}, sigma={lds_sigma})")
        else:
            eff_label_dist = bin_counts.astype(float)
            print(f"ℹ️  LDS disabled: using empirical distribution")
        
        # Avoid division by zero
        eff_label_dist = np.maximum(eff_label_dist, 1e-6)
        
        # Compute inverse weights
        if reweight == 'sqrt_inv':
            weights = 1.0 / np.sqrt(eff_label_dist)
        else:  # inv
            weights = 1.0 / eff_label_dist
        
        # Normalize weights to mean=1.0 (keeps loss scale consistent)
        weights = weights / weights.mean()
        
        self.register_buffer('weights', torch.from_numpy(weights).float())
        
        print(f"\n{'='*70}")
        print(f"SimpleLDSLoss Configuration")
        print(f"{'='*70}")
        print(f"Base loss: {base}" + (f" (beta={smoothl1_beta})" if base == 'smoothl1' else ""))
        print(f"Reweight: {reweight}")
        print(f"Bins: {num_bins}")
        print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"Weight mean: {weights.mean():.3f} (normalized to 1.0)")
        print(f"{'='*70}\n")
    
    def forward(self, pred, target):
        """
        Compute weighted loss.
        
        Args:
            pred: predictions [B] or [B, 1]
            target: ground truth [B] or [B, 1]
        
        Returns:
            scalar loss
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Find bin for each sample
        bin_indices = torch.bucketize(target, self.bin_edges) - 1
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        
        # Get sample weights
        sample_weights = self.weights[bin_indices]
        
        # Compute base loss (element-wise)
        if self.base == 'l1':
            loss = F.l1_loss(pred, target, reduction='none')
        elif self.base == 'smoothl1':
            loss = F.smooth_l1_loss(pred, target, reduction='none', beta=self.smoothl1_beta)
        else:  # mse
            loss = F.mse_loss(pred, target, reduction='none')
        
        # Apply weights and reduce
        weighted_loss = loss * sample_weights
        return weighted_loss.mean()
    
    @torch.no_grad()
    def get_sample_weights(self, target):
        """Get weights for given targets (useful for analysis)."""
        target = target.view(-1)
        bin_indices = torch.bucketize(target, self.bin_edges) - 1
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        return self.weights[bin_indices]


def create_lds_weights(targets, kernel='gaussian', ks=5, sigma=2.0, num_bins=100, reweight='sqrt_inv'):
    """
    Create LDS weights for given targets.
    
    This is a standalone function to compute LDS weights without creating a loss function.
    Useful for custom loss implementations that need LDS reweighting.
    
    Args:
        targets: numpy array or tensor of training labels
        kernel: 'gaussian', 'triang', or 'laplace'
        ks: kernel size (must be odd)
        sigma: bandwidth parameter (None uses default)
        num_bins: number of bins to discretize labels
        reweight: 'inv' (1/count) or 'sqrt_inv' (1/sqrt(count))
    
    Returns:
        weights: numpy array of weights per bin
    """
    # Convert to numpy if needed
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    targets = np.array(targets).flatten()
    
    # Handle default sigma
    if sigma is None:
        sigma = 2.0
    
    # Normalize kernel name
    if kernel == 'laplacian':
        kernel = 'laplace'
    
    # Create bin edges
    y_min, y_max = float(targets.min()), float(targets.max())
    bin_edges = np.linspace(y_min, y_max, num_bins + 1)
    
    # Count samples per bin
    bin_counts, _ = np.histogram(targets, bins=bin_edges)
    
    # Apply LDS smoothing
    kernel_window = get_lds_kernel_window(kernel, ks, sigma)
    eff_label_dist = convolve1d(bin_counts.astype(float), weights=kernel_window, mode='constant')
    
    # Avoid division by zero
    eff_label_dist = np.maximum(eff_label_dist, 1e-6)
    
    # Compute inverse weights
    if reweight == 'sqrt_inv':
        weights = 1.0 / np.sqrt(eff_label_dist)
    else:  # inv
        weights = 1.0 / eff_label_dist
    
    # Normalize weights to mean=1.0
    weights = weights / weights.mean()
    
    return weights
