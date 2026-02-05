"""
Balanced MSE Loss (BMSE) - ICML 2022
=====================================

From paper: "Balanced MSE for Imbalanced Visual Regression"

This loss reweights samples based on label density using a Gaussian Mixture Model (GMM).
Rare samples (in the tail of the distribution) receive higher weights.

Key benefits:
- Principled Bayesian approach to density estimation
- Automatic adaptation to any target distribution
- Proven effective on age estimation, depth prediction, and other imbalanced tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


class BalancedMSELoss(nn.Module):
    """
    Balanced MSE Loss using GMM-based density estimation.
    
    Args:
        y_train: Training targets for fitting the GMM
        n_components: Number of GMM components (default: 8)
        sigma: Smoothing factor for density (default: 1.0)
        min_weight: Minimum weight to prevent numerical issues (default: 1.0)
        max_weight: Maximum weight to prevent extreme values (default: 100.0)
        use_sqrt: Whether to use sqrt of inverse density (softer weighting)
    
    Example:
        >>> y_train = np.array([...])  # Training targets
        >>> criterion = BalancedMSELoss(y_train, n_components=8)
        >>> loss = criterion(predictions, targets)
    """
    
    def __init__(self, y_train, n_components=8, sigma=1.0, 
                 min_weight=1.0, max_weight=100.0, use_sqrt=False):
        super().__init__()
        
        self.sigma = sigma
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_sqrt = use_sqrt
        
        # Fit GMM to training labels
        print(f"[BMSE] Fitting GMM with {n_components} components on {len(y_train)} samples...")
        self.gmm = GaussianMixture(
            n_components=n_components, 
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        self.gmm.fit(y_train.reshape(-1, 1))
        
        # Precompute density statistics for normalization
        log_densities = self.gmm.score_samples(y_train.reshape(-1, 1))
        self.density_mean = log_densities.mean()
        self.density_std = log_densities.std() + 1e-8
        
        # Compute weight statistics for logging
        weights = self._compute_weights(y_train)
        print(f"[BMSE] Weight range: [{weights.min():.2f}, {weights.max():.2f}], mean: {weights.mean():.2f}")
        
    def _compute_weights(self, targets):
        """Compute inverse density weights for targets."""
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy().reshape(-1, 1)
        else:
            targets_np = np.array(targets).reshape(-1, 1)
        
        # Get log density from GMM
        log_density = self.gmm.score_samples(targets_np)
        density = np.exp(log_density)
        
        # Inverse density weighting
        weights = 1.0 / (density * self.sigma + 1e-8)
        
        # Apply sqrt if requested (softer weighting)
        if self.use_sqrt:
            weights = np.sqrt(weights)
        
        # Clip to prevent extreme weights
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        return weights
        
    def forward(self, pred, target):
        """
        Compute balanced MSE loss.
        
        Args:
            pred: Predictions (B,) or (B, 1)
            target: Targets (B,)
            
        Returns:
            Scalar loss value
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Compute weights
        weights = self._compute_weights(target)
        weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
        
        # Weighted MSE
        mse = (pred - target) ** 2
        loss = (weights * mse).mean()
        
        return loss
    
    def get_sample_weights(self, targets):
        """
        Get weights for visualization/analysis.
        
        Args:
            targets: Target values
            
        Returns:
            numpy array of weights
        """
        return self._compute_weights(targets)


class BalancedL1Loss(nn.Module):
    """
    Balanced L1 Loss (MAE) using GMM-based density estimation.
    
    Same principle as BMSE but with L1 loss instead of MSE.
    L1 is more robust to outliers.
    """
    
    def __init__(self, y_train, n_components=8, sigma=1.0, 
                 min_weight=1.0, max_weight=50.0, use_sqrt=True):
        super().__init__()
        
        self.sigma = sigma
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_sqrt = use_sqrt
        
        # Fit GMM to training labels
        print(f"[BL1] Fitting GMM with {n_components} components on {len(y_train)} samples...")
        self.gmm = GaussianMixture(
            n_components=n_components, 
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        self.gmm.fit(y_train.reshape(-1, 1))
        
        # Compute weight statistics for logging
        weights = self._compute_weights(y_train)
        print(f"[BL1] Weight range: [{weights.min():.2f}, {weights.max():.2f}], mean: {weights.mean():.2f}")
        
    def _compute_weights(self, targets):
        """Compute inverse density weights for targets."""
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy().reshape(-1, 1)
        else:
            targets_np = np.array(targets).reshape(-1, 1)
        
        # Get log density from GMM
        log_density = self.gmm.score_samples(targets_np)
        density = np.exp(log_density)
        
        # Inverse density weighting
        weights = 1.0 / (density * self.sigma + 1e-8)
        
        # Apply sqrt if requested (softer weighting)
        if self.use_sqrt:
            weights = np.sqrt(weights)
        
        # Clip to prevent extreme weights
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        return weights
        
    def forward(self, pred, target):
        """Compute balanced L1 loss."""
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Compute weights
        weights = self._compute_weights(target)
        weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
        
        # Weighted L1
        l1 = torch.abs(pred - target)
        loss = (weights * l1).mean()
        
        return loss
    
    def get_sample_weights(self, targets):
        """Get weights for visualization/analysis."""
        return self._compute_weights(targets)


class BMCLoss(nn.Module):
    """
    Balanced MSE via Calibration (BMC).
    
    A simpler version that learns the noise parameter during training.
    From the BMC paper, this estimates the uncertainty in predictions.
    
    Args:
        init_noise_sigma: Initial noise standard deviation (learnable)
    """
    
    def __init__(self, init_noise_sigma=1.0):
        super().__init__()
        # Learnable noise parameter
        self.noise_sigma = nn.Parameter(torch.tensor(init_noise_sigma))
        
    def forward(self, pred, target):
        """
        Compute BMC loss.
        
        Args:
            pred: Predictions (B,) or (B, 1)
            target: Targets (B,)
            
        Returns:
            Scalar loss value
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Squared error
        sq_error = (pred - target) ** 2
        
        # BMC formula: loss = sq_error / (2 * sigma^2) + log(sigma)
        # This balances fitting error with learned uncertainty
        noise_var = self.noise_sigma ** 2
        loss = sq_error / (2 * noise_var) + torch.log(self.noise_sigma)
        
        return loss.mean()
