"""
RankSim Loss - NeurIPS 2022
============================

From paper: "RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression"

This loss regularizes feature representations to preserve target ranking relationships.
If two samples have similar targets, their features should also be similar.

Key benefits:
- Forces model to learn meaningful representations for rare samples
- Contrastive learning approach
- Combines with any base regression loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankSimLoss(nn.Module):
    """
    RankSim Loss: Regularizes feature similarity to match target similarity.
    
    Args:
        temperature: Temperature for softmax similarity scaling (default: 0.1)
        lambda_ranksim: Weight for RankSim regularization term (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-8)
        use_hard_mining: Whether to focus on hard negative pairs (default: False)
    
    Example:
        >>> criterion = RankSimLoss(temperature=0.1, lambda_ranksim=1.0)
        >>> base_loss = F.mse_loss(predictions, targets)
        >>> total_loss = base_loss + criterion(features, targets)
    """
    
    def __init__(self, temperature=0.1, lambda_ranksim=1.0, eps=1e-8, use_hard_mining=False):
        super().__init__()
        self.temperature = temperature
        self.lambda_ranksim = lambda_ranksim
        self.eps = eps
        self.use_hard_mining = use_hard_mining
        
    def _get_target_similarity(self, targets):
        """
        Compute target similarity matrix based on ranking.
        
        Args:
            targets: (B,) target values
            
        Returns:
            (B, B) similarity matrix based on target distances
        """
        # Compute pairwise target distances
        targets = targets.view(-1, 1)
        target_dists = torch.abs(targets - targets.T)  # (B, B)
        
        # Convert distances to similarities
        # Closer targets = higher similarity
        target_sim = torch.exp(-target_dists / self.temperature)
        
        return target_sim
    
    def _get_feature_similarity(self, features):
        """
        Compute feature similarity matrix using cosine similarity.
        
        Args:
            features: (B, D) feature representations
            
        Returns:
            (B, B) cosine similarity matrix
        """
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Cosine similarity
        feature_sim = torch.mm(features_norm, features_norm.T)  # (B, B)
        
        # Scale to [0, 1]
        feature_sim = (feature_sim + 1) / 2
        
        return feature_sim
    
    def forward(self, features, targets, mask=None):
        """
        Compute RankSim regularization loss.
        
        Args:
            features: (B, D) feature representations from model
            targets: (B,) target values
            mask: Optional (B,) binary mask for rare samples (focus on these)
            
        Returns:
            RankSim regularization loss
        """
        B = features.size(0)
        
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Compute similarity matrices
        target_sim = self._get_target_similarity(targets)
        feature_sim = self._get_feature_similarity(features)
        
        # Mask diagonal (self-similarity)
        diag_mask = 1 - torch.eye(B, device=features.device)
        
        # Apply optional sample mask
        if mask is not None:
            # Focus on pairs where at least one sample is rare
            mask = mask.view(-1, 1)
            pair_mask = (mask + mask.T) > 0
            diag_mask = diag_mask * pair_mask.float()
        
        # RankSim loss: MSE between feature and target similarities
        loss = (feature_sim - target_sim) ** 2
        loss = (loss * diag_mask).sum() / (diag_mask.sum() + self.eps)
        
        return self.lambda_ranksim * loss


class RankSimWithBaseLoss(nn.Module):
    """
    Combined loss: Base regression loss + RankSim regularization.
    
    This is the full loss function as proposed in the paper.
    
    Args:
        base_loss: Base regression loss (e.g., MSE, L1, Huber)
        temperature: Temperature for RankSim (default: 0.1)
        lambda_ranksim: Weight for RankSim term (default: 1.0)
        
    Example:
        >>> criterion = RankSimWithBaseLoss('mse', lambda_ranksim=1.0)
        >>> loss = criterion(predictions, features, targets)
    """
    
    def __init__(self, base_loss='mse', temperature=0.1, lambda_ranksim=1.0):
        super().__init__()
        
        # Base loss
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        elif base_loss == 'l1':
            self.base_loss = nn.L1Loss()
        elif base_loss == 'huber':
            self.base_loss = nn.SmoothL1Loss()
        else:
            self.base_loss = base_loss  # Custom loss
            
        self.ranksim = RankSimLoss(
            temperature=temperature, 
            lambda_ranksim=lambda_ranksim
        )
        
    def forward(self, pred, features, target, mask=None):
        """
        Compute combined loss.
        
        Args:
            pred: (B,) or (B, 1) predictions
            features: (B, D) feature representations
            target: (B,) targets
            mask: Optional rare sample mask
            
        Returns:
            Total loss
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Base regression loss
        base_loss = self.base_loss(pred, target)
        
        # RankSim regularization
        ranksim_loss = self.ranksim(features, target, mask)
        
        return base_loss + ranksim_loss


class RankSimFocalLoss(nn.Module):
    """
    RankSim with Focal regression loss for extra emphasis on rare samples.
    
    Combines:
    - Focal-like regression loss (emphasizes hard examples)
    - RankSim regularization (preserves ranking structure)
    """
    
    def __init__(self, gamma=2.0, temperature=0.1, lambda_ranksim=1.0):
        super().__init__()
        self.gamma = gamma
        self.ranksim = RankSimLoss(
            temperature=temperature, 
            lambda_ranksim=lambda_ranksim
        )
        
    def forward(self, pred, features, target, mask=None):
        """Compute focal + RankSim loss."""
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Focal-like regression loss
        errors = torch.abs(pred - target)
        focal_weights = errors.detach() ** self.gamma
        focal_loss = (focal_weights * errors).mean()
        
        # RankSim regularization
        ranksim_loss = self.ranksim(features, target, mask)
        
        return focal_loss + ranksim_loss


def get_rare_sample_mask(targets, threshold_percentile=10):
    """
    Create a binary mask for rare samples.
    
    Uses the sideSlip distribution - samples in the tails are "rare".
    
    Args:
        targets: Target values
        threshold_percentile: Percentile threshold for defining "rare"
        
    Returns:
        Binary mask (1 for rare, 0 for common)
    """
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = targets
    
    import numpy as np
    
    # Compute percentiles
    low = np.percentile(targets_np, threshold_percentile)
    high = np.percentile(targets_np, 100 - threshold_percentile)
    
    # Rare = in the tails
    mask = (targets_np < low) | (targets_np > high)
    
    return torch.tensor(mask.astype(np.float32))


class MultiHeadRankSimLoss(nn.Module):
    """
    Multi-scale RankSim: Uses multiple temperature scales.
    
    This captures both local and global ranking relationships.
    """
    
    def __init__(self, temperatures=[0.05, 0.1, 0.2], lambda_ranksim=1.0):
        super().__init__()
        self.losses = nn.ModuleList([
            RankSimLoss(temperature=t, lambda_ranksim=lambda_ranksim/len(temperatures))
            for t in temperatures
        ])
        
    def forward(self, features, targets, mask=None):
        """Compute multi-scale RankSim loss."""
        total_loss = 0
        for loss_fn in self.losses:
            total_loss = total_loss + loss_fn(features, targets, mask)
        return total_loss
