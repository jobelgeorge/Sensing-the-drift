"""
Variational Autoencoder for Imbalanced Sideslip Regression
============================================================

This module implements VAE architectures for the sideslip angle prediction task.

Key Features:
- RNN-based encoder/decoder for sequence data
- Prediction head for sideslip regression from latent space
- Band-aware contrastive loss for latent separation
- Latent augmentation utilities

Author: MAIT Group 2
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class RNNEncoder(nn.Module):
    """
    RNN Encoder for sequence-to-latent mapping.
    
    Input: (batch, seq_len, input_dim) -> (batch, latent_dim*2)  # μ, log_σ²
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # RNN backbone 
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity='relu',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Project to latent parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence to latent distribution parameters.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # RNN encoding
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)
        
        # Use last timestep
        h = rnn_out[:, -1, :]  # (batch, hidden_dim)
        
        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class RNNDecoder(nn.Module):
    """
    RNN Decoder for latent-to-sequence reconstruction.
    
    Input: (batch, latent_dim) -> (batch, seq_len, input_dim)
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 22,
        seq_len: int = 20,
        num_layers: int = 1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Project latent to hidden
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        
        # RNN for sequence generation
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity='relu',
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to sequence.
        
        Args:
            z: Latent vectors (batch, latent_dim)
            
        Returns:
            x_recon: Reconstructed sequences (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)
        
        # Project latent to hidden
        h = self.fc_latent(z)  # (batch, hidden_dim)
        h = F.relu(h)
        
        # Repeat for sequence length
        h_seq = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        
        # RNN decoding
        rnn_out, _ = self.rnn(h_seq)  # (batch, seq_len, hidden_dim)
        
        # Project to output
        x_recon = self.fc_out(rnn_out)  # (batch, seq_len, output_dim)
        
        return x_recon


class SideslipPredictor(nn.Module):
    """
    Prediction head for sideslip from latent representation.
    
    Matches the winning RNN architecture's head.
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict sideslip from latent.
        
        Args:
            z: Latent vectors (batch, latent_dim)
            
        Returns:
            y_pred: Predicted sideslip (batch, 1)
        """
        h = self.act(self.fc1(z))
        y_pred = self.fc2(h)
        return y_pred


class SideslipVAE(nn.Module):
    """
    Complete VAE for sideslip prediction.
    
    Architecture:
        Input (B, 20, 22) -> Encoder -> z (B, latent_dim)
                                    |-> Decoder -> Recon (B, 20, 22)
                                    |-> Predictor -> sideslip (B, 1)
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 20,
        encoder_layers: int = 1,
        decoder_layers: int = 1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Components
        self.encoder = RNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers
        )
        
        self.decoder = RNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=decoder_layers
        )
        
        self.predictor = SideslipPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with reconstruction and prediction.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            return_all: If True, return all intermediate values
            
        Returns:
            Dictionary with:
                - y_pred: Sideslip prediction
                - x_recon: Reconstructed sequence
                - mu: Latent mean
                - logvar: Latent log variance
                - z: Sampled latent
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Predict
        y_pred = self.predictor(z)
        
        return {
            'y_pred': y_pred,
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Encode to latent space (for visualization/augmentation).
        """
        mu, logvar = self.encoder(x)
        if sample:
            return self.reparameterize(mu, logvar)
        return mu
    
    def predict_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict sideslip from latent (for augmentation).
        """
        return self.predictor(z)
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Loss Functions
# ============================================================================

class WeightedL1Loss(nn.Module):
    """
    Threshold-based weighted L1 loss (from winning solution).
    
    weight = 1 + slope * max(0, |target| - threshold)
    """
    
    def __init__(self, threshold: float = 2.0, slope: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.slope = slope
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        weights = 1.0 + self.slope * torch.clamp(
            torch.abs(target) - self.threshold, min=0
        )
        loss = (torch.abs(pred.squeeze() - target) * weights).mean()
        return loss


class WeightedReconLoss(nn.Module):
    """
    Reconstruction loss with band-based weighting.
    
    Higher weight for few-shot sample reconstructions.
    """
    
    def __init__(self, threshold: float = 3.0, slope: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.slope = slope
    
    def forward(
        self, 
        x_recon: torch.Tensor, 
        x_orig: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_recon: Reconstructed sequences (batch, seq_len, dim)
            x_orig: Original sequences (batch, seq_len, dim)
            targets: Sideslip targets for weight calculation (batch,)
        """
        # Per-sample MSE
        mse_per_sample = ((x_recon - x_orig) ** 2).mean(dim=(1, 2))  # (batch,)
        
        # Weights based on sideslip magnitude
        weights = 1.0 + self.slope * torch.clamp(
            torch.abs(targets) - self.threshold, min=0
        )
        
        # Weighted mean
        loss = (mse_per_sample * weights).mean()
        return loss


class BandContrastiveLoss(nn.Module):
    """
    Contrastive loss to separate bands in latent space.
    
    Pull same-band samples together, push different bands apart.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self, 
        z: torch.Tensor, 
        bands: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: Latent vectors (batch, latent_dim)
            bands: Band labels as integers (0=many, 1=medium, 2=few) (batch,)
        """
        batch_size = z.size(0)
        
        # Normalize latent vectors
        z_norm = F.normalize(z, dim=1)
        
        # Similarity matrix
        sim = torch.mm(z_norm, z_norm.t()) / self.temperature  # (batch, batch)
        
        # Mask for same band
        band_eq = bands.unsqueeze(0) == bands.unsqueeze(1)  # (batch, batch)
        band_eq = band_eq.float()
        
        # Positive: same band (not self)
        mask_pos = band_eq - torch.eye(batch_size, device=z.device)
        
        # Negative: different band
        mask_neg = 1.0 - band_eq
        
        # InfoNCE-style loss
        # For each sample, pull positives and push negatives
        exp_sim = torch.exp(sim)
        
        # Sum of positive similarities
        pos_sum = (exp_sim * mask_pos).sum(dim=1)
        
        # Sum of all similarities (except self)
        all_sum = (exp_sim * (1 - torch.eye(batch_size, device=z.device))).sum(dim=1)
        
        # Avoid division by zero
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        
        # Only compute for samples with positives
        valid = mask_pos.sum(dim=1) > 0
        if valid.sum() > 0:
            return loss[valid].mean()
        return torch.tensor(0.0, device=z.device)


class VAELoss(nn.Module):
    """
    Combined VAE loss for sideslip prediction.
    
    L_total = L_pred + α·L_recon + β·L_KL + γ·L_contrastive
    """
    
    def __init__(
        self,
        alpha_recon: float = 1.0,
        beta_kl: float = 0.1,
        gamma_contrast: float = 0.0,
        pred_threshold: float = 2.0,
        pred_slope: float = 1.0,
        recon_threshold: float = 3.0,
        recon_slope: float = 0.5
    ):
        super().__init__()
        
        self.alpha = alpha_recon
        self.beta = beta_kl
        self.gamma = gamma_contrast
        
        self.pred_loss = WeightedL1Loss(threshold=pred_threshold, slope=pred_slope)
        self.recon_loss = WeightedReconLoss(threshold=recon_threshold, slope=recon_slope)
        self.contrast_loss = BandContrastiveLoss() if gamma_contrast > 0 else None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        x_orig: torch.Tensor,
        targets: torch.Tensor,
        bands: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total VAE loss.
        
        Args:
            outputs: Dictionary from VAE forward pass
            x_orig: Original input sequences
            targets: Sideslip targets
            bands: Optional band labels for contrastive loss
            
        Returns:
            total_loss: Combined loss value
            components: Dictionary of individual loss components
        """
        # Prediction loss
        l_pred = self.pred_loss(outputs['y_pred'], targets)
        
        # Reconstruction loss
        l_recon = self.recon_loss(outputs['x_recon'], x_orig, targets)
        
        # KL divergence
        mu = outputs['mu']
        logvar = outputs['logvar']
        l_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Contrastive loss (optional)
        l_contrast = torch.tensor(0.0, device=x_orig.device)
        if self.gamma > 0 and bands is not None and self.contrast_loss is not None:
            l_contrast = self.contrast_loss(outputs['z'], bands)
        
        # Total loss
        total = l_pred + self.alpha * l_recon + self.beta * l_kl + self.gamma * l_contrast
        
        components = {
            'loss_total': total.item(),
            'loss_pred': l_pred.item(),
            'loss_recon': l_recon.item(),
            'loss_kl': l_kl.item(),
            'loss_contrast': l_contrast.item()
        }
        
        return total, components


# ============================================================================
# Latent Space Utilities
# ============================================================================

def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two latent vectors.
    
    Better than linear interpolation for high-dimensional spaces.
    """
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    
    omega = torch.acos((z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1))
    sin_omega = torch.sin(omega)
    
    # Handle case where vectors are nearly parallel
    if sin_omega.abs().min() < 1e-6:
        return (1 - t) * z1 + t * z2
    
    s1 = torch.sin((1 - t) * omega) / sin_omega
    s2 = torch.sin(t * omega) / sin_omega
    
    # Interpolate on unit sphere, then scale
    z1_scale = torch.norm(z1, dim=-1, keepdim=True)
    z2_scale = torch.norm(z2, dim=-1, keepdim=True)
    scale = (1 - t) * z1_scale + t * z2_scale
    
    return (s1 * z1_norm + s2 * z2_norm) * scale


def latent_augmentation(
    z_few: torch.Tensor,
    n_aug: int,
    method: str = 'interpolate',
    noise_std: float = 0.1
) -> torch.Tensor:
    """
    Generate augmented latent vectors from few-shot samples.
    
    Args:
        z_few: Latent vectors of few-shot samples (N, latent_dim)
        n_aug: Number of augmented samples to generate
        method: 'interpolate', 'gaussian', or 'both'
        noise_std: Standard deviation for gaussian noise
        
    Returns:
        z_aug: Augmented latent vectors (n_aug, latent_dim)
    """
    n_few = z_few.size(0)
    device = z_few.device
    
    aug_samples = []
    
    if method in ['interpolate', 'both']:
        # Random pairs for interpolation
        n_interp = n_aug // 2 if method == 'both' else n_aug
        for _ in range(n_interp):
            idx1, idx2 = torch.randint(0, n_few, (2,))
            t = torch.rand(1).item()
            z_interp = slerp(z_few[idx1:idx1+1], z_few[idx2:idx2+1], t)
            aug_samples.append(z_interp)
    
    if method in ['gaussian', 'both']:
        # Gaussian noise around few-shot samples
        n_gauss = n_aug - len(aug_samples)
        for _ in range(n_gauss):
            idx = torch.randint(0, n_few, (1,)).item()
            noise = torch.randn_like(z_few[idx:idx+1]) * noise_std
            z_noisy = z_few[idx:idx+1] + noise
            aug_samples.append(z_noisy)
    
    if len(aug_samples) > 0:
        return torch.cat(aug_samples, dim=0)
    return z_few[:0]  # Empty tensor with correct shape


def classify_band_torch(
    targets: torch.Tensor,
    many_threshold: float = 2.4,
    few_threshold: float = 4.0
) -> torch.Tensor:
    """
    Classify sideslip values into bands (for contrastive loss).
    
    Returns:
        bands: Integer labels (0=many, 1=medium, 2=few)
    """
    abs_targets = torch.abs(targets)
    bands = torch.zeros_like(targets, dtype=torch.long)
    bands[abs_targets > many_threshold] = 1  # medium
    bands[abs_targets > few_threshold] = 2   # few
    return bands


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("Testing VAE components...\n")
    
    # Test shapes
    batch_size = 32
    seq_len = 20
    input_dim = 22
    latent_dim = 16
    
    # Create model
    model = SideslipVAE(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=latent_dim,
        seq_len=seq_len
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size)
    
    outputs = model(x)
    
    print(f"\nForward pass shapes:")
    print(f"  Input: {x.shape}")
    print(f"  y_pred: {outputs['y_pred'].shape}")
    print(f"  x_recon: {outputs['x_recon'].shape}")
    print(f"  mu: {outputs['mu'].shape}")
    print(f"  logvar: {outputs['logvar'].shape}")
    print(f"  z: {outputs['z'].shape}")
    
    # Test loss
    loss_fn = VAELoss(alpha_recon=1.0, beta_kl=0.1, gamma_contrast=0.1)
    bands = classify_band_torch(y)
    
    total_loss, components = loss_fn(outputs, x, y, bands)
    
    print(f"\nLoss components:")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")
    
    # Test augmentation
    z_few = torch.randn(10, latent_dim)
    z_aug = latent_augmentation(z_few, n_aug=20, method='both')
    print(f"\nAugmentation: {z_few.shape} -> {z_aug.shape}")
    
    print("\n✅ All tests passed!")
