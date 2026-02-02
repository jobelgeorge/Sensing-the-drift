"""WeightedL1 Loss - Winning loss function for imbalanced regression."""

import torch
import torch.nn as nn


class WeightedL1Loss(nn.Module):
    """Threshold-based weighted L1. Best: threshold=2.0, slope=1.0."""
    
    def __init__(self, threshold=2.0, slope=1.0):
        super().__init__()
        self.threshold = threshold
        self.slope = slope
    
    def forward(self, pred, target):
        target = target.view_as(pred)
        weights = 1.0 + self.slope * torch.clamp(torch.abs(target) - self.threshold, min=0)
        return (weights * torch.abs(pred - target)).mean()
