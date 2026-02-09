# VAE for Vehicle Sideslip Estimation

Variational Autoencoder experiments for deep imbalanced regression.

## Notebooks
- `01_vae_architecture_sweep.ipynb` - Latent dim, beta, architecture sweeps
- `02_vae_loss_experiments.ipynb` - DIR losses (LDS, Focal, BMSE)
- `03_latent_augmentation.ipynb` - Generate synthetic data via latent space

## Kaggle Results Summary

| Model | Submission File | Kaggle Score |
|-------|-----------------|--------------|
| Vanilla RNN (Baseline) | - | 0.21751 |
| **Best VAE Single** | submission_beta_0.005.csv | **0.25319** |
| Best VAE Ensemble | submission_ensemble_v4.csv | 0.26493 |

## Optimal Configuration

| Parameter | Value |
|-----------|-------|
| latent_dim | 16 |
| hidden_dim | 64 |
| beta_kl | 0.005 |
| alpha_recon | 0.0 |
| gamma_contrast | 0.0 |
| Training | Two-stage (L1 -> WeightedL1) |

## Hyperparameter Sweeps - Kaggle Results

### Latent Dimension
| Dim | Submission File | Kaggle Score |
|-----|-----------------|--------------|
| 8 | submission_latent_dim_8.csv | 0.33069 |
| **16** | submission_latent_dim_16.csv | **0.30004** |
| 32 | submission_latent_dim_32.csv | 0.31325 |
| 64 | submission_latent_dim_64.csv | 0.32478 |

### KL Weight (beta)
| Beta | Submission File | Kaggle Score |
|------|-----------------|--------------|
| 0.001 | submission_beta_0.001.csv | 0.33198 |
| **0.005** | submission_beta_0.005.csv | **0.25319** |
| 0.01 | submission_beta_kl_0.01.csv | 0.28298 |
| 0.02 | submission_beta_0.02.csv | 0.30602 |
| 0.1 | submission_beta_kl_0.1.csv | 0.29411 |
| 0.5 | submission_beta_kl_0.5.csv | 0.28700 |
| 1.0 | submission_beta_kl_1.0.csv | 0.29270 |

### Architecture
| Architecture | Submission File | Kaggle Score |
|--------------|-----------------|--------------|
| Baseline | submission_arch_baseline.csv | 0.29591 |
| Wide | submission_arch_wide.csv | 0.27342 |
| **Deep** | submission_arch_deep.csv | **0.26613** |
| Large | submission_arch_large.csv | 0.31155 |
| Small | submission_arch_small.csv | 0.34495 |
| Deep 2L-128 | submission_deep_2layer_128.csv | 0.26502 |

### Loss Functions (Alpha Recon / Gamma Contrast)
| Config | Submission File | Kaggle Score |
|--------|-----------------|--------------|
| **alpha=0.0** | submission_alpha_recon_0.0.csv | **0.28368** |
| alpha=0.1 | submission_alpha_recon_0.1.csv | 0.33590 |
| alpha=0.5 | submission_alpha_recon_0.5.csv | 0.31966 |
| alpha=1.0 | submission_alpha_recon_1.0.csv | 0.29705 |
| alpha=2.0 | submission_alpha_recon_2.0.csv | 0.34049 |
| **gamma=0.0** | submission_gamma_contrast_0.0.csv | **0.26752** |
| gamma=0.1 | submission_gamma_contrast_0.1.csv | 0.31221 |
| gamma=0.5 | submission_gamma_contrast_0.5.csv | 0.28130 |
| gamma=1.0 | submission_gamma_contrast_1.0.csv | 0.29306 |

### DIR Losses (LDS/FDS/Focal)
| Loss | Submission File | Kaggle Score |
|------|-----------------|--------------|
| **LDS V1** | predictions_LDS_V1.csv | **0.28695** |
| LDS V2 | predictions_LDS_V2.csv | 0.36409 |
| LDS V3 | predictions_LDS_V3.csv | 0.45150 |
| LDS V4 | predictions_LDS_V4.csv | 0.32742 |
| Focal 1.0 | submission_Focal_1.0.csv | 0.31616 |
| Focal 3.0 | submission_Focal_3.0.csv | 0.39786 |
| FDS V1 | predictions_FDS_V1.csv | 0.61806 |
| FDS V2 | predictions_FDS_V2.csv | 0.48196 |
| FDS V3 | predictions_FDS_V3.csv | 0.62249 |
| FDS V4 | predictions_FDS_V4.csv | 0.44314 |

### Ensembles
| Ensemble | Submission File | Kaggle Score |
|----------|-----------------|--------------|
| **V4** | submission_ensemble_v4.csv | **0.26493** |
| Optimal | submission_ensemble_optimal.csv | 0.26692 |
| Top3 | submission_ensemble_top3_overall.csv | 0.27775 |
| Weighted | submission_ensemble_weighted.csv | 0.36272 |
| All | submission_ensemble_all.csv | 0.38072 |

### Data Augmentation
| Method | Submission File | Kaggle Score |
|--------|-----------------|--------------|
| **No Augmentation** | submission_baseline_no_aug.csv | **0.29615** |
| Extended Soft Aug | submission_extended_soft_aug.csv | 0.32850 |
| Mixup α=0.2 | submission_mixup_alpha_0.2.csv | 0.47659 |

## Hypotheses Tested

| Experiment | Result | Evidence |
|------------|--------|----------|
| Higher latent dims | 16 optimal | 0.30004 vs 0.32478 (dim=64) |
| KL regularization | 0.005 best | 0.25319 vs 0.33198 (beta=0.001) |
| Reconstruction loss | Hurts prediction | alpha=0.0 (0.28368) vs alpha=2.0 (0.34049) |
| Contrast loss | Hurts prediction | gamma=0.0 (0.26752) vs gamma=1.0 (0.29306) |
| DIR losses (LDS, FDS, Focal) | All degraded | LDS best=0.28695, FDS worst=0.62249 |
| Data augmentation | All hurt | No aug (0.29615) vs Mixup (0.47659) |

## Conclusion

VAE underperforms baseline RNN by +16.5% error (0.25319 vs 0.21751). 

**Key findings:**
- Best VAE config: `beta=0.005`, no reconstruction/contrast loss
- DIR losses (LDS, FDS, Focal) all degrade performance
- Data augmentation consistently hurts
- Ensembling provides marginal improvement (0.26493)
- Simple RNN remains superior for this task
