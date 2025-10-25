import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class FrequencyBasedWeightedLoss(nn.Module):
    """
    Imbalanced regression loss based on target VALUE DISTRIBUTION.
    
    Divides dataset into frequency-based buckets (many/medium/few-shot)
    and applies inverse frequency weighting to prioritize rare values.
    
    Key difference from magnitude-based: Thresholds are computed from
    actual data distribution (percentiles), not arbitrary magnitudes.
    
    Args:
        train_targets: numpy array or tensor of ALL training targets
        percentiles: (low, high) defining splits, e.g., (90, 99) means:
                     - Many-shot: [0, 90th percentile]
                     - Medium-shot: (90th, 99th percentile]
                     - Few-shot: (99th, 100th percentile]
        base_weights: Manual boost factors (default inverse frequency)
        use_huber: Whether to use Huber loss (robust) or L1
        huber_beta: Huber loss parameter
        focal_gamma: Optional focal loss gamma for hard example mining
    """
    def __init__(
        self,
        train_targets: np.ndarray,
        percentiles: Tuple[float, float] = (90.0, 99.0),
        base_weights: Optional[Tuple[float, float, float]] = None,
        use_huber: bool = True,
        huber_beta: float = 0.1,
        focal_gamma: float = 0.0,  # 0 = disabled, 1-2 = moderate focus
        eps: float = 1e-8
    ):
        super().__init__()
        
        # Convert to numpy for percentile computation
        if isinstance(train_targets, torch.Tensor):
            train_targets = train_targets.cpu().numpy()
        
        train_targets = train_targets.flatten()
        
        # Compute ABSOLUTE value distribution (sideslip can be +/-)
        abs_targets = np.abs(train_targets)
        
        # Define thresholds from DATA DISTRIBUTION
        t0 = float(np.percentile(abs_targets, percentiles[0]))
        t1 = float(np.percentile(abs_targets, percentiles[1]))
        
        print(f"\n{'='*60}")
        print(f"FrequencyBasedWeightedLoss Configuration:")
        print(f"{'='*60}")
        print(f"Percentile splits: {percentiles}")
        print(f"  Many-shot threshold (90th %ile):  |y| ≤ {t0:.4f}")
        print(f"  Medium-shot threshold (99th %ile): {t0:.4f} < |y| ≤ {t1:.4f}")
        print(f"  Few-shot threshold:                |y| > {t1:.4f}")
        
        # Count samples per bucket
        n_many = np.sum(abs_targets <= t0)
        n_medium = np.sum((abs_targets > t0) & (abs_targets <= t1))
        n_few = np.sum(abs_targets > t1)
        total = len(abs_targets)
        
        print(f"\nData Distribution:")
        print(f"  Many-shot:   {n_many:6d} samples ({100*n_many/total:.1f}%)")
        print(f"  Medium-shot: {n_medium:6d} samples ({100*n_medium/total:.1f}%)")
        print(f"  Few-shot:    {n_few:6d} samples ({100*n_few/total:.1f}%)")
        
        # Compute inverse frequency weights
        if base_weights is None:
            # Effective sample weighting: weight_i = 1 / n_i
            # Normalized so many-shot gets weight 1.0
            w_many = 1.0
            w_medium = n_many / max(n_medium, 1)
            w_few = n_many / max(n_few, 1)
            
            # Optional: sqrt to reduce extreme weights
            w_medium = np.sqrt(w_medium)
            w_few = np.sqrt(w_few)
            
            base_weights = (w_many, w_medium, w_few)
        
        self.register_buffer("t0", torch.tensor(t0))
        self.register_buffer("t1", torch.tensor(t1))
        self.w_many, self.w_medium, self.w_few = [float(w) for w in base_weights]
        
        print(f"\nBucket Weights:")
        print(f"  Many-shot:   {self.w_many:.2f}x")
        print(f"  Medium-shot: {self.w_medium:.2f}x")
        print(f"  Few-shot:    {self.w_few:.2f}x")
        print(f"{'='*60}\n")
        
        self.use_huber = use_huber
        self.huber_beta = huber_beta
        self.focal_gamma = focal_gamma
        self.eps = eps
        
        if use_huber:
            self._base_loss = nn.SmoothL1Loss(beta=huber_beta, reduction="none")
        else:
            self._base_loss = nn.L1Loss(reduction="none")
    
    def bucket_masks(self, target: torch.Tensor):
        """
        Returns boolean masks for each frequency bucket based on |target|.
        
        Returns:
            (mask_many, mask_medium, mask_few): each shape [B]
        """
        target_abs = target.abs()
        mask_many = (target_abs <= self.t0)
        mask_medium = (target_abs > self.t0) & (target_abs <= self.t1)
        mask_few = (target_abs > self.t1)
        return mask_many, mask_medium, mask_few
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss with frequency-based buckets.
        
        Args:
            pred: predictions [B] or [B, 1]
            target: ground truth [B] or [B, 1]
        
        Returns:
            Scalar loss
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        
        # Base loss (L1 or Huber)
        base_loss = self._base_loss(pred, target)  # [B]
        
        # Get frequency bucket masks
        mask_many, mask_medium, mask_few = self.bucket_masks(target)
        
        # Apply bucket weights
        weights = torch.zeros_like(base_loss)
        weights = torch.where(mask_many, weights + self.w_many, weights)
        weights = torch.where(mask_medium, weights + self.w_medium, weights)
        weights = torch.where(mask_few, weights + self.w_few, weights)
        
        # Optional: Focal loss modulation for hard examples
        if self.focal_gamma > 0:
            # Focal weight: (1 - pt)^gamma where pt ~ 1 for easy samples
            # Use normalized error as proxy: high error → hard sample
            with torch.no_grad():
                normalized_error = base_loss / (base_loss.max() + self.eps)
                focal_weight = normalized_error ** self.focal_gamma
            weights = weights * (1.0 + focal_weight)
        
        # Weighted average
        weighted_loss = (weights * base_loss).sum() / (weights.sum() + self.eps)
        
        return weighted_loss
    
    @torch.no_grad()
    def bucket_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute MAE per frequency bucket for monitoring.
        
        Returns:
            dict with keys: mae_many, mae_medium, mae_few, count_many, etc.
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        
        mask_many, mask_medium, mask_few = self.bucket_masks(target)
        
        def compute_mae(mask):
            if mask.any():
                return (pred[mask] - target[mask]).abs().mean().item()
            return float('nan')
        
        return {
            'mae_many': compute_mae(mask_many),
            'mae_medium': compute_mae(mask_medium),
            'mae_few': compute_mae(mask_few),
            'count_many': mask_many.sum().item(),
            'count_medium': mask_medium.sum().item(),
            'count_few': mask_few.sum().item(),
        }
    
    @torch.no_grad()
    def analyze_predictions(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Detailed analysis for debugging model behavior in each bucket.
        
        Returns:
            dict with statistics per bucket
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        
        mask_many, mask_medium, mask_few = self.bucket_masks(target)
        
        stats = {}
        for name, mask in [('many', mask_many), ('medium', mask_medium), ('few', mask_few)]:
            if mask.any():
                p = pred[mask]
                t = target[mask]
                error = (p - t).abs()
                
                stats[name] = {
                    'mae': error.mean().item(),
                    'rmse': (error ** 2).mean().sqrt().item(),
                    'max_error': error.max().item(),
                    'median_error': error.median().item(),
                    'count': mask.sum().item(),
                    'target_range': (t.min().item(), t.max().item()),
                    'pred_range': (p.min().item(), p.max().item()),
                }
            else:
                stats[name] = None
        
        return stats