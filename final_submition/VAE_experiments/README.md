# VAE for Vehicle Sideslip Estimation

Variational Autoencoder experiments for deep imbalanced regression.

## Notebooks
- `01_vae_architecture_sweep.ipynb` - Latent dim, beta, architecture sweeps
- `02_vae_loss_experiments.ipynb` - DIR losses (LDS, Focal, BMSE)
- `03_latent_augmentation.ipynb` - Generate synthetic data via latent space

## Results

| Model | Params | Score | Status |
|-------|--------|-------|--------|
| Vanilla RNN (Baseline) | 13K | 0.217 | Best |
| Best VAE Single | ~50K | 0.253 | +16% error |
| VAE Ensemble | 150K | 0.265 | +22% error |

## Optimal Configuration

| Parameter | Value |
|-----------|-------|
| latent_dim | 16 |
| hidden_dim | 64 |
| beta_kl | 0.005 |
| alpha_recon | 0.0 |
| gamma_contrast | 0.0 |
| Training | Two-stage (L1 -> WeightedL1) |

## Hyperparameter Sweeps

### Latent Dimension
| Dim | Score |
|-----|-------|
| 8 | 0.331 |
| 16 | 0.253 |
| 32 | 0.313 |
| 64 | 0.325 |

### KL Weight (beta)
| beta | Score |
|------|-------|
| 0.001 | 0.332 |
| 0.005 | 0.253 |
| 0.01 | 0.283 |
| 0.02 | 0.306 |

### Loss Functions
| Loss | Score |
|------|-------|
| WeightedL1 | 0.253 |
| L1 | 0.260 |
| LDS | 0.268 |
| Focal | 0.275 |
| BMC | 0.290 |

## Hypotheses Tested

| Experiment | Result |
|------------|--------|
| Higher latent dims | 16 optimal, larger overfit |
| KL regularization | 0.005 best (20x smaller than default) |
| Reconstruction loss | Hurts prediction |
| DIR losses (LDS, FDS, Focal) | All failed or degraded |

## Conclusion

VAE underperforms baseline RNN by 16.6%. Generative modeling overhead doesn't benefit imbalanced regression. Simple RNN is sufficient.
