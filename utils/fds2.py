import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d


class FDS(nn.Module):
    """
    Stable Feature Distribution Smoothing (FDS) for Imbalanced Regression
    """
    
    def __init__(
        self,
        feature_dim,
        bucket_num=100,
        bucket_start=-15.0,
        bucket_end=15.0,
        start_smooth=10,
        sigma=2.0,
        momentum=0.95,
        calibration_weight=0.1
    ):
        super(FDS, self).__init__()
        
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.bucket_end = bucket_end
        self.start_smooth = start_smooth
        self.sigma = sigma
        self.momentum = momentum
        self.calibration_weight = calibration_weight
        
        # Create bucket boundaries
        self.bucket_width = (bucket_end - bucket_start) / bucket_num
        
        # Running statistics
        self.register_buffer(
            'running_mean',
            torch.zeros(bucket_num, feature_dim)
        )
        self.register_buffer(
            'running_var',
            torch.ones(bucket_num, feature_dim)
        )
        self.register_buffer(
            'running_count',
            torch.zeros(bucket_num)
        )
        
        # Smoothed statistics
        self.register_buffer(
            'smoothed_mean',
            torch.zeros(bucket_num, feature_dim)
        )
        
        self.initialized = False
        self.min_samples = 10
    
    def _get_bucket_idx(self, y):
        """Map target values to bucket indices"""
        y = y.view(-1).float()
        bucket_idx = ((y - self.bucket_start) / self.bucket_width).long()
        bucket_idx = torch.clamp(bucket_idx, 0, self.bucket_num - 1)
        return bucket_idx
    
    def update_running_stats(self, features, y):
        """Update running statistics with variance tracking"""
        bucket_idx = self._get_bucket_idx(y)
        
        with torch.no_grad():
            for i in range(self.bucket_num):
                mask = (bucket_idx == i)
                n_samples = mask.sum().item()
                
                if n_samples > 0:
                    bucket_features = features[mask].detach()
                    bucket_mean = bucket_features.mean(dim=0)
                    
                    # 🔹 FIX: Only compute variance if we have 2+ samples
                    if n_samples > 1:
                        bucket_var = bucket_features.var(dim=0, unbiased=True) + 1e-6
                    else:
                        bucket_var = torch.ones_like(bucket_mean) * 1e-6
                    
                    if self.running_count[i] == 0:
                        self.running_mean[i] = bucket_mean
                        self.running_var[i] = bucket_var
                    else:
                        self.running_mean[i] = (
                            self.momentum * self.running_mean[i] +
                            (1 - self.momentum) * bucket_mean
                        )
                        self.running_var[i] = (
                            self.momentum * self.running_var[i] +
                            (1 - self.momentum) * bucket_var
                        )
                    
                    self.running_count[i] += n_samples
    
    def smooth_statistics(self):
        """Apply Gaussian smoothing with boundary handling"""
        running_mean_np = self.running_mean.cpu().numpy()
        smoothed_mean_np = np.zeros_like(running_mean_np)
        
        for i in range(self.feature_dim):
            # Only smooth if we have enough data
            valid_buckets = self.running_count.cpu().numpy() > 0
            if valid_buckets.sum() > 5:
                smoothed_mean_np[:, i] = gaussian_filter1d(
                    running_mean_np[:, i],
                    sigma=self.sigma,
                    mode='nearest'
                )
            else:
                smoothed_mean_np[:, i] = running_mean_np[:, i]
        
        self.smoothed_mean = torch.from_numpy(smoothed_mean_np).to(
            self.running_mean.device
        ).float()
        
        self.initialized = True
    
    def forward(self, features, y, epoch):
        """Apply stable FDS calibration"""
        # Always update statistics during training
        if self.training:
            self.update_running_stats(features, y)
        
        # Only apply calibration after warmup
        if epoch < self.start_smooth:
            return features
        
        # Smooth statistics periodically
        if not self.initialized or epoch == self.start_smooth or epoch % 5 == 0:
            self.smooth_statistics()
        
        # Apply calibration with safety checks
        bucket_idx = self._get_bucket_idx(y)
        calibrated_features = features.clone()
        
        for i in range(self.bucket_num):
            mask = (bucket_idx == i)
            
            # Only calibrate if we have enough samples
            if mask.sum() > 0 and self.running_count[i] >= self.min_samples:
                delta = self.smoothed_mean[i] - self.running_mean[i]
                
                # Clip delta to prevent extreme calibration
                delta = torch.clamp(delta, -2.0, 2.0)
                
                # Apply weighted calibration
                calibrated_features[mask] = (
                    features[mask] + self.calibration_weight * delta
                )
        
        return calibrated_features