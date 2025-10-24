# WeightedHybridLoss.py
import torch
import torch.nn as nn
from typing import Optional, Tuple

class WeightedHybridLoss(nn.Module):
    """
    Magnitude-aware regression loss with per-bucket weights.
    - Computes either L1 or Huber (SmoothL1) *then* applies magnitude bucket weights.
    - Buckets are defined on the **original-scale** target |y| using thresholds.
    - Works whether your y is already scaled or not (pass a scaler if needed).

    Args:
        thresholds: tuple of two floats (t0, t1) defining 3 buckets on |y_orig|
                    [0, t0], (t0, t1], (t1, +inf).
        weights:    tuple of three floats, per-bucket weights (w0, w1, w2).
        use_huber:  if True uses SmoothL1 (Huber-like); otherwise plain L1.
        huber_beta: beta parameter for SmoothL1.
        scaler_y:   optional object with inverse_transform(np_like) -> original scale.
                    If you use e.g. sklearn StandardScaler on y, pass it here.
        target_is_scaled: set True if incoming y is scaled and needs inverse transform.
    """
    def __init__(
        self,
        thresholds: Tuple[float, float] = (0.1, 0.4),
        weights: Tuple[float, float, float] = (1.0, 2.0, 4.0),
        use_huber: bool = True,
        huber_beta: float = 0.1,
        scaler_y: Optional[object] = None,
        target_is_scaled: bool = False,
        eps: float = 1e-12
    ):
        super().__init__()
        assert thresholds[0] < thresholds[1], "thresholds must be (t0 < t1)"
        self.register_buffer("t0", torch.tensor(float(thresholds[0])))
        self.register_buffer("t1", torch.tensor(float(thresholds[1])))
        self.w0, self.w1, self.w2 = [float(w) for w in weights]

        self.use_huber = use_huber
        self.huber_beta = float(huber_beta)
        self.scaler_y = scaler_y
        self.target_is_scaled = target_is_scaled
        self.eps = eps

        if self.use_huber:
            self._base = nn.SmoothL1Loss(beta=self.huber_beta, reduction="none")
        else:
            self._base = nn.L1Loss(reduction="none")

    # ----- utilities -----
    def _to_original(self, y: torch.Tensor) -> torch.Tensor:
        """
        If target_is_scaled=True and a scaler is provided, invert the scaling.
        Accepts torch tensor; converts to cpu().numpy() for sklearn-like scalers,
        then returns a torch tensor on the original device.
        """
        if not self.target_is_scaled or self.scaler_y is None:
            return y

        # handle [B] or [B,1]
        y_ = y.detach()
        was_unsqueezed = False
        if y_.ndim == 1:
            y_ = y_.unsqueeze(-1)
            was_unsqueezed = True

        # sklearn inverse_transform expects numpy
        y_np = y_.cpu().numpy()
        y_inv_np = self.scaler_y.inverse_transform(y_np)
        y_inv = torch.from_numpy(y_inv_np).to(y.device, y.dtype)

        if was_unsqueezed:
            y_inv = y_inv.squeeze(-1)

        return y_inv

    def bucket_masks(self, y: torch.Tensor):
        """
        Build boolean masks per bucket, based on |y_orig|.
        Returns (m0, m1, m2), each shape [B].
        """
        y_orig_abs = self._to_original(y).abs()
        m0 = (y_orig_abs <= self.t0)                                       # [0, t0]
        m1 = (y_orig_abs > self.t0) & (y_orig_abs <= self.t1)              # (t0, t1]
        m2 = (y_orig_abs > self.t1)                                        # (t1, +inf)
        return m0, m1, m2

    # ----- main forward -----
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: shape [B] or [B,1]
        Returns scalar loss.
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)

        base = self._base(pred, target)  # [B]

        m0, m1, m2 = self.bucket_masks(target)
        # avoid all-false masks by adding tiny eps mass (numerically stable)
        w = torch.zeros_like(base)
        w = torch.where(m0, w + self.w0, w)
        w = torch.where(m1, w + self.w1, w)
        w = torch.where(m2, w + self.w2, w)

        loss = (w * base).sum() / (w.sum() + self.eps)
        return loss

    # ----- helpful metrics for logging -----
    @torch.no_grad()
    def bucket_mae(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Returns MAE per bucket on the **model scale** (i.e., whatever is fed in),
        which is fine for *relative* tracking. If you want MAE on original scale,
        inverse-transform pred/target before calling this.
        """
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        m0, m1, m2 = self.bucket_masks(target)

        def mae(mask):
            if mask.any():
                return (pred[mask] - target[mask]).abs().mean().item()
            return float("nan")

        return {
            "mae_b0": mae(m0),
            "mae_b1": mae(m1),
            "mae_b2": mae(m2),
        }
