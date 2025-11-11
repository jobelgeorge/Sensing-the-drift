"""
Simple Focal-R Loss for Imbalanced Regression
==============================================

Based on: "Delving into Deep Imbalanced Regression" (ICML 2021)
But SIMPLIFIED - only the Focal-R component (no LDS, no manual buckets)

Focal-R: Focuses on hard examples by upweighting samples with high errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalRLoss(nn.Module):
    """
    Focal-R Loss: Focus on hard examples in regression.
    
    Standard Loss: L = MAE(pred, target)
    Focal-R Loss:  L = (error / max_error)^γ × MAE(pred, target)
    
    Where γ controls focusing:
    - γ = 0: No focusing (standard MAE)
    - γ = 1: Moderate focusing (recommended)
    - γ = 2: Strong focusing on hard examples
    
    Args:
        gamma: Focusing parameter (default: 1.0)
        beta: SmoothL1 beta parameter (default: 0.1, ignored if base='l1')
        base: Base loss ('l1', 'smoothl1', or 'mse')
    """
    
    def __init__(self, gamma=1.0, beta=0.1, base='smoothl1'):
        super().__init__()
        
        assert base in ['l1', 'smoothl1', 'mse'], f"Unsupported base: {base}"
        
        self.gamma = gamma
        self.beta = beta
        self.base = base
        
        print(f"\n{'='*70}")
        print(f"[Focal-R Loss Configuration]")
        print(f"{'='*70}")
        print(f"Base loss: {base}" + (f" (beta={beta})" if base == 'smoothl1' else ""))
        print(f"Gamma (focusing): {gamma}")
        print(f"  γ=0: No focusing (standard {base.upper()})")
        print(f"  γ=1: Moderate focusing ✅")
        print(f"  γ=2: Strong focusing on hard examples")
        print(f"{'='*70}\n")
    
    def forward(self, pred, target):
        """
        Compute Focal-R Loss.
        
        Args:
            pred: Predictions (B, 1) or (B,)
            target: Ground truth (B, 1) or (B,)
        
        Returns:
            Scalar loss
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute base loss (element-wise, no reduction)
        if self.base == 'l1':
            base_loss = F.l1_loss(pred, target, reduction='none')
        elif self.base == 'smoothl1':
            base_loss = F.smooth_l1_loss(pred, target, reduction='none', beta=self.beta)
        else:  # mse
            base_loss = F.mse_loss(pred, target, reduction='none')
        
        # Focal modulation: upweight hard examples
        if self.gamma > 0:
            with torch.no_grad():
                # Normalize errors by max error in batch
                max_error = base_loss.max().clamp_min(1e-6)
                focal_weight = (base_loss / max_error).pow(self.gamma)
        else:
            focal_weight = torch.ones_like(base_loss)
        
        # Apply focal weighting
        focal_loss = focal_weight * base_loss
        
        return focal_loss.mean()
    
    def __repr__(self):
        return f"FocalRLoss(gamma={self.gamma}, beta={self.beta}, base='{self.base}')"


class SimpleLDSLoss(nn.Module):
    """
    Simplified LDS Loss (no smoothing, just inverse density reweighting).
    Useful as a baseline to compare with DIR-LDS.
    """
    
    def __init__(self, train_targets, base='smoothl1', beta=0.1, num_bins=1000):
        super().__init__()
        
        import numpy as np
        
        self.base = base
        self.beta = beta
        
        # Create bins
        y_min, y_max = train_targets.min(), train_targets.max()
        bin_edges = np.linspace(y_min, y_max, num_bins + 1)
        self.register_buffer('bin_edges', torch.from_numpy(bin_edges).float())
        
        # Compute histogram (NO SMOOTHING)
        hist, _ = np.histogram(train_targets, bins=bin_edges)
        hist = hist + 1e-6  # Avoid division by zero
        
        # Inverse density weights
        weights = 1.0 / hist
        weights = weights / weights.mean()  # Normalize
        
        self.register_buffer('weights', torch.from_numpy(weights).float())
        
        print(f"\n{'='*70}")
        print(f"[Simple LDS Loss - No Smoothing]")
        print(f"{'='*70}")
        print(f"Base loss: {base}" + (f" (beta={beta})" if base == 'smoothl1' else ""))
        print(f"Bins: {num_bins}")
        print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"⚠️  No Gaussian smoothing (raw inverse density)")
        print(f"{'='*70}\n")
    
    def forward(self, pred, target):
        """Compute weighted loss."""
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Find bin indices
        bin_indices = torch.bucketize(target, self.bin_edges) - 1
        bin_indices = bin_indices.clamp(0, len(self.weights) - 1)
        
        # Get weights
        sample_weights = self.weights[bin_indices]
        
        # Compute base loss
        if self.base == 'l1':
            loss = F.l1_loss(pred, target, reduction='none')
        elif self.base == 'smoothl1':
            loss = F.smooth_l1_loss(pred, target, reduction='none', beta=self.beta)
        else:  # mse
            loss = F.mse_loss(pred, target, reduction='none')
        
        # Apply weighting
        weighted_loss = loss * sample_weights
        
        return weighted_loss.mean()
