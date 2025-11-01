"""
Simple Bucket-Weighted Loss
No LDS, No Focal - Just clean bucket weighting
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class SimpleBucketLoss(nn.Module):
    """
    Dead simple bucket-weighted loss.
    No smoothing, no focal - just pure bucket prioritization.
    """
    
    def __init__(
        self,
        train_targets,
        bucket_weights=(1.0, 1.0, 1.0),
        dist_csv_path=None,
        use_huber=True,
        huber_beta=0.1
    ):
        super().__init__()
        
        self.bucket_weights = torch.tensor(bucket_weights, dtype=torch.float32)
        self.use_huber = use_huber
        self.huber_beta = huber_beta
        
        # Load distribution to get bucket boundaries
        if dist_csv_path:
            dist_df = pd.read_csv(dist_csv_path)
            
            # Get boundaries from distribution
            many_mask = dist_df['band'] == 'many'
            medium_mask = dist_df['band'] == 'medium'
            few_mask = dist_df['band'] == 'few'
            
            many_max = dist_df[many_mask]['sideSlip'].abs().max()
            medium_min = dist_df[medium_mask]['sideSlip'].abs().min()
            medium_max = dist_df[medium_mask]['sideSlip'].abs().max()
            few_min = dist_df[few_mask]['sideSlip'].abs().min()
            
            self.many_threshold = many_max
            self.medium_threshold = medium_max
            
            print(f"\n{'='*80}")
            print(f"SimpleBucketLoss Configuration")
            print(f"{'='*80}")
            print(f"Bucket Thresholds:")
            print(f"  Many-shot:   |y| ≤ {self.many_threshold:.2f}°")
            print(f"  Medium-shot: {medium_min:.2f}° < |y| ≤ {self.medium_threshold:.2f}°")
            print(f"  Few-shot:    |y| > {few_min:.2f}°")
            print(f"\nBucket Weights: {bucket_weights}")
            print(f"Loss Type: {'Huber' if use_huber else 'L1'}")
            if use_huber:
                print(f"Huber Beta: {huber_beta}")
            print(f"{'='*80}\n")
        else:
            # Default thresholds if no CSV
            self.many_threshold = 0.1
            self.medium_threshold = 0.4
    
    def forward(self, pred, target):
        """
        Simple bucket-weighted loss.
        
        Args:
            pred: Model predictions [batch_size, 1]
            target: Ground truth [batch_size, 1]
        
        Returns:
            Weighted loss (scalar)
        """
        # Move weights to same device
        if self.bucket_weights.device != pred.device:
            self.bucket_weights = self.bucket_weights.to(pred.device)
        
        # Compute base loss (element-wise)
        if self.use_huber:
            # Huber loss (smooth L1)
            diff = torch.abs(pred - target)
            loss_elementwise = torch.where(
                diff < self.huber_beta,
                0.5 * diff ** 2 / self.huber_beta,
                diff - 0.5 * self.huber_beta
            )
        else:
            # Pure L1
            loss_elementwise = torch.abs(pred - target)
        
        # Assign bucket weights based on |target|
        target_abs = torch.abs(target)
        
        weights = torch.ones_like(target)
        
        # Many-shot: |y| <= many_threshold
        many_mask = target_abs <= self.many_threshold
        weights[many_mask] = self.bucket_weights[0]
        
        # Medium-shot: many_threshold < |y| <= medium_threshold
        medium_mask = (target_abs > self.many_threshold) & (target_abs <= self.medium_threshold)
        weights[medium_mask] = self.bucket_weights[1]
        
        # Few-shot: |y| > medium_threshold
        few_mask = target_abs > self.medium_threshold
        weights[few_mask] = self.bucket_weights[2]
        
        # Apply weights and average
        weighted_loss = loss_elementwise * weights
        return weighted_loss.mean()
    
    def bucket_metrics(self, pred, target):
        """
        Calculate MAE for each bucket.
        
        Args:
            pred: Predictions [N, 1]
            target: Ground truth [N, 1]
        
        Returns:
            dict with bucket MAEs
        """
        target_abs = torch.abs(target)
        errors = torch.abs(pred - target)
        
        # Many-shot
        many_mask = target_abs <= self.many_threshold
        if many_mask.sum() > 0:
            mae_many = errors[many_mask].mean().item()
            count_many = many_mask.sum().item()
        else:
            mae_many = 0.0
            count_many = 0
        
        # Medium-shot
        medium_mask = (target_abs > self.many_threshold) & (target_abs <= self.medium_threshold)
        if medium_mask.sum() > 0:
            mae_medium = errors[medium_mask].mean().item()
            count_medium = medium_mask.sum().item()
        else:
            mae_medium = 0.0
            count_medium = 0
        
        # Few-shot
        few_mask = target_abs > self.medium_threshold
        if few_mask.sum() > 0:
            mae_few = errors[few_mask].mean().item()
            count_few = few_mask.sum().item()
        else:
            mae_few = 0.0
            count_few = 0
        
        return {
            'mae_many': mae_many,
            'mae_medium': mae_medium,
            'mae_few': mae_few,
            'count_many': count_many,
            'count_medium': count_medium,
            'count_few': count_few
        }
