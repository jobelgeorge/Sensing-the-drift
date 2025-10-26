"""
Enhanced LDS + Focal-R Loss for Imbalanced Regression
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d


class EnhancedLDS_FocalR_Loss(nn.Module):
    """
    LDS (smoothed inverse density) * (1 + Focal^gamma) * explicit bucket weights
    """
    def __init__(
        self,
        train_targets: np.ndarray,
        bucket_weights: tuple = (1.0, 4.0, 12.0),
        lds_kernel: str = 'gaussian',
        lds_ks: int = 5,
        lds_sigma: float = 2.0,
        focal_gamma: float = 2.0,
        use_huber: bool = True,
        huber_beta: float = 0.05,
        num_bins: int = 100,
        eps: float = 1e-8
    ):
        super().__init__()

        if isinstance(train_targets, torch.Tensor):
            train_targets = train_targets.cpu().numpy()
        train_targets = train_targets.flatten()
        abs_targets = np.abs(train_targets)

        # 90th / 99th percentile buckets (many / medium / few)
        self.t0 = float(np.percentile(abs_targets, 90.0))
        self.t1 = float(np.percentile(abs_targets, 99.0))

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

        # nice to print once for visibility
        n_many = np.sum(abs_targets <= self.t0)
        n_medium = np.sum((abs_targets > self.t0) & (abs_targets <= self.t1))
        n_few = np.sum(abs_targets > self.t1)
        total = len(abs_targets)
        print(f"[EnhancedLDS+FocalR] many≤{self.t0:.4f}, med≤{self.t1:.4f} | "
              f"dist: many={n_many/total:.1%}, med={n_medium/total:.1%}, few={n_few/total:.1%} | "
              f"weights: {self.w_many}/{self.w_medium}/{self.w_few} | "
              f"focal_gamma={focal_gamma}, huber_beta={huber_beta}")

    def _compute_lds_weights(self, targets, kernel, ks, sigma, num_bins):
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
        if target_value in self.weights_dict:
            return self.weights_dict[target_value]

        keys = np.array(sorted(self.weights_dict.keys()))  # ensure sorted
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        base = self._base_loss(pred, target)

        # LDS
        lds_w = torch.tensor([self._get_lds_weight(t.item()) for t in target],
                             device=pred.device, dtype=pred.dtype)

        # Focal-R
        if self.focal_gamma > 0:
            with torch.no_grad():
                max_err = base.max().clamp_min(1e-6)
                foc = (base / max_err).pow(self.focal_gamma)
        else:
            foc = torch.ones_like(base)

        # Buckets
        abs_t = target.abs()
        bucket = torch.ones_like(target)
        bucket[abs_t <= self.t0] = self.w_many
        bucket[(abs_t > self.t0) & (abs_t <= self.t1)] = self.w_medium
        bucket[abs_t > self.t1] = self.w_few

        weights = lds_w * (1.0 + foc) * bucket
        return (weights * base).sum() / (weights.sum() + self.eps)

    @torch.no_grad()
    def bucket_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        p = pred.squeeze(-1)
        t = target.squeeze(-1)
        abs_t = t.abs()
        mask_many = (abs_t <= self.t0)
        mask_med = (abs_t > self.t0) & (abs_t <= self.t1)
        mask_few = (abs_t > self.t1)

        def mae(m): return (p[m] - t[m]).abs().mean().item() if m.any() else float('nan')
        return dict(mae_many=mae(mask_many), mae_medium=mae(mask_med), mae_few=mae(mask_few))
