"""
Enhanced LDS + Focal-R Loss for Imbalanced Regression

HYPERPARAMETER TUNING GUIDE:
================================
1. bucket_weights: (many, medium, few)
   - Start: (1.0, 2.0, 4.0)
   - If few-shot MAE > 1.0: increase to (1.0, 3.0, 8.0)
   - If many-shot MAE > 0.19: decrease to (1.0, 1.5, 3.0)

2. focal_gamma: Controls hard example mining
   - 0.5: Gentle focus on hard samples
   - 1.0: Moderate focus (recommended start)
   - 2.0: Aggressive focus (use if few-shot stuck)

3. lds_sigma: Controls label smoothing
   - 1.0: Minimal smoothing
   - 2.0: Recommended for high imbalance (>100x)
   - 3.0: Maximum smoothing (use if overfitting few-shot)

4. huber_beta: Robustness parameter
   - 0.05: More robust to outliers
   - 0.1: Balanced
   - 0.2: Less robust but faster convergence
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # ✅ ADD: For CSV loading
from scipy.ndimage import gaussian_filter1d


class EnhancedLDS_FocalR_Loss(nn.Module):
    """
    LDS + Focal-R with CSV-based bucket definitions (competition-aligned).
    """
    def __init__(
        self,
        train_targets: np.ndarray,
        bucket_weights: tuple = (1.0, 3.0, 8.0),
        lds_kernel: str = 'gaussian',
        lds_ks: int = 7,
        lds_sigma: float = 2.0,
        focal_gamma: float = 1.0,
        use_huber: bool = True,
        huber_beta: float = 0.08,
        num_bins: int = 100,
        dist_csv_path: str = None,  # ✅ NEW: Path to distribution CSV
        eps: float = 1e-8
    ):
        super().__init__()

        if isinstance(train_targets, torch.Tensor):
            train_targets = train_targets.cpu().numpy()
        train_targets = train_targets.flatten()

        # ============================================================================
        # ✅ FIX: Load bucket definitions from CSV (competition-aligned)
        # ============================================================================
        if dist_csv_path is None:
            raise ValueError(
                "dist_csv_path is required! Pass path to sideSlip_distribution.csv"
            )
        
        dist_df = pd.read_csv(dist_csv_path)
        self.shot_band_mapping = dist_df.set_index("sideSlip")["band"].to_dict()
        
        # Build reverse mapping: band → list of sideslip values
        self.band_ranges = {
            'many': set(),
            'medium': set(),
            'few': set()
        }
        for sideslip, band in self.shot_band_mapping.items():
            self.band_ranges[band].add(float(sideslip))
        
        # Compute approximate boundaries (for logging only)
        many_vals = [abs(s) for s, b in self.shot_band_mapping.items() if b == 'many']
        medium_vals = [abs(s) for s, b in self.shot_band_mapping.items() if b == 'medium']
        
        self.t0 = max(many_vals) if many_vals else 2.4  # Many-shot max
        self.t1 = max(medium_vals) if medium_vals else 5.0  # Medium-shot max
        
        self.w_many, self.w_medium, self.w_few = bucket_weights
        self.focal_gamma = focal_gamma
        self.use_huber = use_huber
        self.huber_beta = huber_beta
        self.eps = eps

        self.weights_dict = self._compute_lds_weights(
            train_targets, lds_kernel, lds_ks, lds_sigma, num_bins
        )

        self._base_loss = nn.SmoothL1Loss(beta=huber_beta, reduction="none") if use_huber \
                          else nn.L1Loss(reduction="none")

        # Print configuration (using CSV-based stats)
        y_rounded = np.round(train_targets, 1)
        bands_train = pd.Series(y_rounded).map(self.shot_band_mapping).fillna("unknown")
        band_counts = bands_train.value_counts()
        
        n_many = band_counts.get('many', 0)
        n_medium = band_counts.get('medium', 0)
        n_few = band_counts.get('few', 0)
        total = len(train_targets)
        
        print(f"\n{'='*70}")
        print(f"[EnhancedLDS+FocalR Loss - CSV-Aligned Configuration]")
        print(f"{'='*70}")
        print(f"Loaded {len(self.shot_band_mapping)} band definitions from CSV")
        print(f"Approximate boundaries (max |sideslip| per band):")
        print(f"  Many-shot:   <= {self.t0:.4f} degrees")
        print(f"  Medium-shot: <= {self.t1:.4f} degrees")
        print(f"Data distribution:")
        print(f"  Many:   {n_many:6d} ({n_many/total*100:5.2f}%)")
        print(f"  Medium: {n_medium:6d} ({n_medium/total*100:5.2f}%)")
        print(f"  Few:    {n_few:6d} ({n_few/total*100:5.2f}%)")
        print(f"Bucket weights: {self.w_many:.1f} / {self.w_medium:.1f} / {self.w_few:.1f}")
        print(f"Focal gamma: {focal_gamma} | Huber beta: {huber_beta}")
        print(f"LDS: kernel={lds_kernel}, sigma={lds_sigma}")
        print(f"{'='*70}\n")

    def _compute_lds_weights(self, targets, kernel, ks, sigma, num_bins):
        """
        Compute LDS (label distribution smoothing) weights for targets
        """
        tmin, tmax = targets.min(), targets.max()
        bin_edges = np.linspace(tmin, tmax, num_bins + 1)

        bin_idx = np.digitize(targets, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)

        counts = np.bincount(bin_idx, minlength=num_bins).astype(float)
        
        if kernel == 'gaussian':
            smooth = gaussian_filter1d(counts, sigma=sigma, mode='constant', cval=0.0)
        else:
            smooth = np.convolve(counts, np.bartlett(ks), mode='same')
        
        smooth = np.maximum(smooth, 1.0)
        inv = 1.0 / smooth
        inv /= inv.mean()
        
        wdict = {}
        for t in targets:
            i = np.digitize([t], bin_edges)[0] - 1
            i = np.clip(i, 0, num_bins - 1)
            wdict[float(t)] = float(inv[i])
        return wdict

    def _get_lds_weight(self, target_value):
        """
        Get LDS weight for a specific target value (interpolated if not exact)
        """
        if target_value in self.weights_dict:
            return self.weights_dict[target_value]
        
        keys = np.array(sorted(self.weights_dict.keys()))
        idx = np.searchsorted(keys, target_value)
        
        if idx == 0:
            return self.weights_dict[keys[0]]
        elif idx == len(keys):
            return self.weights_dict[keys[-1]]
        else:
            k1, k2 = keys[idx-1], keys[idx]
            w1, w2 = self.weights_dict[k1], self.weights_dict[k2]
            alpha = (target_value - k1) / (k2 - k1 + self.eps)
            return (1 - alpha) * w1 + alpha * w2

    def _get_bucket(self, target_value: float) -> str:
        """
        ✅ NEW: Map target value to band using CSV lookup.
        """
        rounded = round(target_value, 1)
        return self.shot_band_mapping.get(rounded, "unknown")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass (loss value) for the given predictions and targets
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        base = self._base_loss(pred, target)

        # LDS weights
        lds_w = torch.tensor([self._get_lds_weight(t.item()) for t in target],
                             device=pred.device, dtype=pred.dtype)

        # Focal-R modulation
        if self.focal_gamma > 0:
            with torch.no_grad():
                max_err = base.max().clamp_min(1e-6)
                foc = (base / max_err).pow(self.focal_gamma)
        else:
            foc = torch.ones_like(base)

        # ✅ FIX: Bucket weights using CSV lookup
        bucket = torch.ones_like(target)
        for i, t in enumerate(target):
            band = self._get_bucket(t.item())
            if band == 'many':
                bucket[i] = self.w_many
            elif band == 'medium':
                bucket[i] = self.w_medium
            elif band == 'few':
                bucket[i] = self.w_few
            # else: unknown → weight=1.0

        # Combined weighting
        weights = lds_w * (1.0 + foc) * bucket
        return (weights * base).sum() / (weights.sum() + self.eps)

    @torch.no_grad()
    def bucket_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """
        ✅ FIX: Compute MAE per bucket using CSV-based bands.
        """
        p = pred.squeeze(-1)
        t = target.squeeze(-1)
        
        # Map targets to bands
        bands = [self._get_bucket(val.item()) for val in t]
        bands_tensor = torch.tensor(
            [1 if b=='many' else 2 if b=='medium' else 3 if b=='few' else 0 for b in bands],
            device=p.device
        )
        
        mask_many = (bands_tensor == 1)
        mask_med = (bands_tensor == 2)
        mask_few = (bands_tensor == 3)

        def mae(m): 
            return (p[m] - t[m]).abs().mean().item() if m.any() else float('nan')
        
        return dict(
            mae_many=mae(mask_many), 
            mae_medium=mae(mask_med), 
            mae_few=mae(mask_few)
        )
