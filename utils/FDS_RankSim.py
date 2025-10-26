"""
FDS (Feature Distribution Smoothing) + RankSim regularization
Works with any model exposing: forward(x, return_features=True) -> (pred, feat)
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d


class FDS_Module(nn.Module):
    def __init__(self, feature_dim: int, num_bins: int = 100, lds_sigma: float = 2.0,
                 start_update_epoch: int = 1, start_smooth_epoch: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.lds_sigma = lds_sigma
        self.start_update_epoch = start_update_epoch
        self.start_smooth_epoch = start_smooth_epoch

        self.current_epoch = 0
        self.register_buffer('running_mean', torch.zeros(num_bins, feature_dim))
        self.register_buffer('running_var', torch.ones(num_bins, feature_dim))
        self.register_buffer('running_count', torch.zeros(num_bins))
        self.register_buffer('smoothed_mean', torch.zeros(num_bins, feature_dim))
        self.register_buffer('smoothed_var', torch.ones(num_bins, feature_dim))

        self.statistics_computed = False
        self.min_target = None
        self.max_target = None

    def _get_bin_idx(self, targets):
        if self.min_target is None or self.max_target is None:
            return torch.zeros_like(targets, dtype=torch.long)
        norm = (targets - self.min_target) / (self.max_target - self.min_target + 1e-8)
        idx = (norm * self.num_bins).long()
        return torch.clamp(idx, 0, self.num_bins - 1)

    @torch.no_grad()
    def update_running_stats(self, features, targets):
        if self.current_epoch < self.start_update_epoch:
            return
        if self.min_target is None:
            self.min_target = targets.min().item()
            self.max_target = targets.max().item()
        else:
            self.min_target = min(self.min_target, targets.min().item())
            self.max_target = max(self.max_target, targets.max().item())

        bin_idx = self._get_bin_idx(targets)

        for b in range(self.num_bins):
            m = (bin_idx == b)
            if m.any():
                f = features[m]
                n = f.size(0)
                old_n = self.running_count[b].item()
                new_n = old_n + n

                mean_f = f.mean(dim=0)
                var_f = f.var(dim=0, unbiased=False)

                if old_n == 0:
                    self.running_mean[b] = mean_f
                    self.running_var[b] = var_f
                else:
                    delta = mean_f - self.running_mean[b]
                    self.running_mean[b] = self.running_mean[b] + delta * (n / new_n)
                    self.running_var[b] = (
                        (old_n * self.running_var[b] + n * var_f + (n * old_n / new_n) * delta ** 2) / new_n
                    )
                self.running_count[b] = new_n

    @torch.no_grad()
    def compute_smoothed_statistics(self):
        mean_np = self.running_mean.cpu().numpy()
        var_np = self.running_var.cpu().numpy()

        s_mean = np.zeros_like(mean_np)
        s_var = np.zeros_like(var_np)
        for d in range(self.feature_dim):
            s_mean[:, d] = gaussian_filter1d(mean_np[:, d], sigma=self.lds_sigma, mode='nearest')
            s_var[:, d] = gaussian_filter1d(var_np[:, d], sigma=self.lds_sigma, mode='nearest')

        self.smoothed_mean = torch.from_numpy(s_mean).to(self.running_mean.device)
        self.smoothed_var = torch.from_numpy(s_var).to(self.running_var.device)
        self.statistics_computed = True

    def forward(self, features, targets):
        if self.current_epoch < self.start_smooth_epoch:
            return features
        if not self.statistics_computed:
            self.compute_smoothed_statistics()

        bin_idx = self._get_bin_idx(targets)
        calibrated = torch.empty_like(features)
        for i, b in enumerate(bin_idx):
            rm = self.running_mean[b]
            rs = torch.sqrt(self.running_var[b].clamp_min(1e-8))
            sm = self.smoothed_mean[b]
            ss = torch.sqrt(self.smoothed_var[b].clamp_min(1e-8))
            calibrated[i] = (features[i] - rm) / rs * ss + sm
        return calibrated

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)


class RankSimLoss(nn.Module):
    """Encourage nearby targets to have more similar features than far targets."""
    def __init__(self, margin: float = 0.2, temperature: float = 0.1, sample_ratio: float = 0.5):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.sample_ratio = sample_ratio

    def forward(self, features, targets):
        B = features.size(0)
        if B < 3:
            return features.new_tensor(0.0)

        if self.sample_ratio < 1.0:
            k = max(3, int(B * self.sample_ratio))
            idx = torch.randperm(B, device=features.device)[:k]
            features = features[idx]
            targets = targets[idx]
            B = k

        f = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        tdiff = (targets.unsqueeze(0) - targets.unsqueeze(1)).abs()
        sim = torch.mm(f, f.t()) / self.temperature

        loss = features.new_zeros(())
        count = 0
        for i in range(B):
            mask = torch.ones(B, dtype=torch.bool, device=features.device)
            mask[i] = False
            d = tdiff[i][mask]
            s = sim[i][mask]
            if d.numel() < 2:
                continue
            near = d.argmin()
            far = d.argmax()
            loss = loss + torch.relu(self.margin + s[far] - s[near])
            count += 1
        return loss / max(1, count)


class FDS_RankSim_Wrapper(nn.Module):
    """
    Wrap any regressor:
      total_loss = main_loss + λ * RankSim(calibrated_features)
    """
    def __init__(self, model: nn.Module, main_criterion: nn.Module,
                 fds_feature_dim: int = 128, ranksim_weight: float = 0.05):
        super().__init__()
        self.model = model
        self.criterion = main_criterion
        self.fds = FDS_Module(feature_dim=fds_feature_dim,
                              num_bins=100, lds_sigma=2.0,
                              start_update_epoch=1, start_smooth_epoch=10)
        self.ranksim = RankSimLoss(margin=0.2, temperature=0.1, sample_ratio=0.5)
        self.ranksim_weight = ranksim_weight

    def forward(self, x, targets, update_stats=True):
        targets = targets.squeeze(-1)
        preds, feats = self.model(x, return_features=True)
        preds = preds.squeeze(-1)

        if self.training and update_stats:
            with torch.no_grad():
                self.fds.update_running_stats(feats.detach(), targets)

        cal_feats = self.fds(feats, targets)
        main_loss = self.criterion(preds.unsqueeze(-1), targets.unsqueeze(-1))
        ranksim_loss = self.ranksim(cal_feats, targets)
        total = main_loss + self.ranksim_weight * ranksim_loss
        return total, preds.unsqueeze(-1), main_loss, ranksim_loss

    def set_epoch(self, epoch: int):
        self.fds.set_epoch(epoch)

    # make optimizers work on the base model params
    def parameters(self, recurse: bool = True):
        return self.model.parameters()
