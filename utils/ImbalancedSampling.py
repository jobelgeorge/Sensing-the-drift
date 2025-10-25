import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Optional, Tuple


class FrequencyBucketSampler(Sampler):
    """
    Custom sampler that OVERSAMPLES rare target values (few-shot region).
    
    Adapted for sequential_dataset which uses sliding windows.
    Ensures each batch contains a balanced mix of many/medium/few-shot samples.
    
    Args:
        dataset: PyTorch dataset (sequential_dataset or Subset of it)
        percentiles: (low, high) for bucket splits, e.g., (90, 99)
        bucket_probs: Sampling probability per bucket (many, medium, few)
                      Default: (0.3, 0.3, 0.4) = 40% few-shot
        replacement: Sample with replacement (required for oversampling)
        num_samples: Total samples per epoch (default: 2x dataset size)
    """
    def __init__(
        self,
        dataset,
        percentiles: Tuple[float, float] = (90.0, 99.0),
        bucket_probs: Tuple[float, float, float] = (0.3, 0.3, 0.4),
        replacement: bool = True,
        num_samples: Optional[int] = None
    ):
        # Handle both sequential_dataset and Subset (from random_split)
        # We'll extract targets by directly iterating through the dataset
        self.dataset_length = len(dataset)
        
        print(f"Extracting targets from dataset (type: {type(dataset).__name__})...")
        print(f"Dataset length: {self.dataset_length}")
        
        # Extract all targets by iterating through the dataset
        # This works for both Subset and direct sequential_dataset
        targets = []
        for i in range(self.dataset_length):
            _, y = dataset[i]  # Get (x, y) pair
            if isinstance(y, torch.Tensor):
                targets.append(y.item())
            else:
                targets.append(float(y))
        
        targets = np.array(targets)
        print(f"Extracted {len(targets)} targets")
        print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        print(f"Target mean: {targets.mean():.4f}, std: {targets.std():.4f}")
        
        # Compute ABSOLUTE value distribution (sideslip can be +/-)
        abs_targets = np.abs(targets)
        
        # Define thresholds from DATA DISTRIBUTION
        t0 = float(np.percentile(abs_targets, percentiles[0]))
        t1 = float(np.percentile(abs_targets, percentiles[1]))
        
        print(f"\n{'='*70}")
        print(f"FrequencyBucketSampler Configuration:")
        print(f"{'='*70}")
        print(f"Dataset type: {type(dataset).__name__}")
        print(f"Dataset length: {self.dataset_length}")
        print(f"Percentile splits: {percentiles}")
        print(f"  Many-shot threshold (90th %ile):  |y| ≤ {t0:.4f}")
        print(f"  Medium-shot threshold (99th %ile): {t0:.4f} < |y| ≤ {t1:.4f}")
        print(f"  Few-shot threshold:                |y| > {t1:.4f}")
        
        # Assign bucket indices (indices into the dataset, not raw data)
        # These are 0 to len(dataset)-1
        bucket_masks = {
            'many': abs_targets <= t0,
            'medium': (abs_targets > t0) & (abs_targets <= t1),
            'few': abs_targets > t1
        }
        
        self.bucket_indices = {
            'many': np.where(bucket_masks['many'])[0],
            'medium': np.where(bucket_masks['medium'])[0],
            'few': np.where(bucket_masks['few'])[0]
        }
        
        # Count samples per bucket
        n_many = len(self.bucket_indices['many'])
        n_medium = len(self.bucket_indices['medium'])
        n_few = len(self.bucket_indices['few'])
        total = n_many + n_medium + n_few
        
        print(f"\nData Distribution:")
        print(f"  Many-shot:   {n_many:6d} samples ({100*n_many/total:.1f}%)")
        print(f"  Medium-shot: {n_medium:6d} samples ({100*n_medium/total:.1f}%)")
        print(f"  Few-shot:    {n_few:6d} samples ({100*n_few/total:.1f}%)")
        
        self.bucket_probs = bucket_probs
        self.replacement = replacement
        self.num_samples = num_samples or (2 * self.dataset_length)
        
        print(f"\nSampling Strategy:")
        print(f"  Many-shot probability:   {bucket_probs[0]:.1%}")
        print(f"  Medium-shot probability: {bucket_probs[1]:.1%}")
        print(f"  Few-shot probability:    {bucket_probs[2]:.1%}")
        print(f"  Samples per epoch: {self.num_samples} (vs original {self.dataset_length})")
        print(f"  Oversampling factor: {self.num_samples/self.dataset_length:.1f}x")
        print(f"{'='*70}\n")
        
        # Validate buckets are not empty
        for name, indices in self.bucket_indices.items():
            if len(indices) == 0:
                print(f"WARNING: {name} bucket is EMPTY! Adjust percentiles.")
    
    def __iter__(self):
        """Generate indices for one epoch with controlled bucket sampling."""
        # Decide which bucket each sample comes from
        bucket_choices = np.random.choice(
            ['many', 'medium', 'few'],
            size=self.num_samples,
            p=self.bucket_probs
        )
        
        # Sample indices from chosen buckets
        indices = []
        for bucket in bucket_choices:
            bucket_idx = self.bucket_indices[bucket]
            if len(bucket_idx) > 0:
                # Sample one index from this bucket
                idx = np.random.choice(bucket_idx, size=1, replace=self.replacement)[0]
                indices.append(int(idx))
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class MixupAugmentation:
    """
    Mixup data augmentation for regression (Zhang et al., 2017).
    
    Particularly effective for imbalanced regression:
    - Creates synthetic samples between existing points
    - Smooths decision boundaries in rare regions
    - Acts as strong regularization
    
    Usage:
        mixup = MixupAugmentation(alpha=0.2, prob=0.5)
        
        # In training loop:
        x_mixed, y_mixed = mixup(x_batch, y_batch)
        pred = model(x_mixed)
        loss = criterion(pred, y_mixed)
    
    Args:
        alpha: Beta distribution parameter (0.2-0.4 typical)
               Higher = more mixing between samples
        prob: Probability of applying mixup to a batch
        prioritize_rare: If True, bias toward mixing rare samples
        rare_threshold: |y| threshold to consider "rare"
    """
    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.5,
        prioritize_rare: bool = False,
        rare_threshold: float = 0.4
    ):
        self.alpha = alpha
        self.prob = prob
        self.prioritize_rare = prioritize_rare
        self.rare_threshold = rare_threshold
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Apply mixup augmentation.
        
        Args:
            x: Input tensor [B, T, F] (batch, time, features)
            y: Target tensor [B, 1] or [B]
        
        Returns:
            (x_mixed, y_mixed): Augmented batch
        """
        if np.random.rand() > self.prob:
            return x, y  # No mixup
        
        batch_size = x.size(0)
        
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Generate random permutation for mixing pairs
        if self.prioritize_rare:
            # Bias toward pairing with rare samples
            y_abs = y.abs().squeeze()
            rare_mask = y_abs > self.rare_threshold
            if rare_mask.any():
                # Higher probability to select rare samples
                weights = torch.where(rare_mask, 
                                     torch.ones_like(y_abs) * 3.0,
                                     torch.ones_like(y_abs))
                weights = weights / weights.sum()
                index = torch.multinomial(weights, batch_size, replacement=True)
            else:
                index = torch.randperm(batch_size, device=x.device)
        else:
            index = torch.randperm(batch_size, device=x.device)
        
        # Mix inputs and targets
        x_mixed = lam * x + (1 - lam) * x[index]
        y_mixed = lam * y + (1 - lam) * y[index]
        
        return x_mixed, y_mixed


class TemporalAugmentation:
    """
    Time-series specific augmentation for LSTM inputs.
    
    Techniques:
    - Jittering: Add small Gaussian noise
    - Magnitude scaling: Simulate sensor noise/drift
    - Time masking: Random dropout of time steps
    
    Helps model generalize to rare dynamic patterns in few-shot region.
    
    Args:
        noise_prob: Probability of adding jitter
        noise_std: Std dev of Gaussian noise (relative to data scale)
        mask_prob: Probability of time step masking
        mask_ratio: Fraction of steps to mask
        scale_prob: Probability of magnitude scaling
        scale_range: (min, max) scaling factors
    """
    def __init__(
        self,
        noise_prob: float = 0.5,
        noise_std: float = 0.02,
        mask_prob: float = 0.2,
        mask_ratio: float = 0.1,
        scale_prob: float = 0.3,
        scale_range: Tuple[float, float] = (0.9, 1.1)
    ):
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.scale_prob = scale_prob
        self.scale_range = scale_range
    
    def add_jitter(self, x: torch.Tensor):
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def magnitude_scale(self, x: torch.Tensor):
        """Scale magnitude by random factor."""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale
    
    def time_mask(self, x: torch.Tensor):
        """Randomly mask time steps (set to 0)."""
        B, T, F = x.shape
        num_mask = int(T * self.mask_ratio)
        
        x = x.clone()  # Don't modify in-place
        for b in range(B):
            mask_indices = np.random.choice(T, size=num_mask, replace=False)
            x[b, mask_indices, :] = 0
        
        return x
    
    def __call__(self, x: torch.Tensor):
        """
        Apply augmentations stochastically.
        
        Args:
            x: Input tensor [B, T, F] (batch, time, features)
        
        Returns:
            Augmented tensor [B, T, F]
        """
        if np.random.rand() < self.noise_prob:
            x = self.add_jitter(x)
        
        if np.random.rand() < self.scale_prob:
            x = self.magnitude_scale(x)
        
        if np.random.rand() < self.mask_prob:
            x = self.time_mask(x)
        
        return x