# Weighted Sampling for Vehicle Sideslip Estimation

Imbalanced regression for vehicle sideslip angle estimation using **sample reweighting** and **ensembling strategies** on  vehicle dynamics datasets.

This notebooks evaluates **Weighted Sampler**, **Weighted Sampler with Seed Ensemble**, and **Weighted Sampler and Snapshot ensemble** on a highly imbalanced dataset.
---

## Models

### Weighted Sampler
A regression model trained with **band-aware sample weights** derived from the sideslip distribution.  
Rare (**few-shot**) samples receive higher importance during training.

### Weighted Sampler + Snapshot Ensemble
Uses **multiple checkpoints from a single training run** (saved at different epochs) and averages their predictions to reduce variance.

### Weighted Sampler + Seed Ensemble
Trains **multiple independent models with different random 10 seeds** and averages predictions to improve robustness.

---

## Architecture

- **Backbone:** Vanilla RNN (ReLU)  
- **Input:** 20 timesteps × 22 features  
- **Output:** Predicted sideslip angle (deterministic)  
- **Loss:** Weighted regression loss  
- **Imbalance Handling:** Sample-wise weighting using `sideSlip_distribution.csv`

---

## Submissions & Scores

| Method | Public Score |
|-------|-------------:|
| Weighted Sampler | **0.25540 (Best)** |
| Weighted Sampler + Snapshot Ensemble | 0.26166 |
| Weighted Sampler + Seed Ensemble | 0.33273 |

---

## Results Summary

| Method | Observation |
|-------|-------------|
| Weighted Sampler | Strong baseline, best overall performance |
| Snapshot Ensemble | Slight degradation, limited diversity between snapshots |
| Seed Ensemble | Significant performance drop due to instability |

---

## Key Findings

- **Weighted sampling is highly effective** for this dataset:
  - Directly addresses label imbalance  
  - Preserves extreme sideslip samples  

- **Snapshot ensembling provides limited benefit:**
  - Snapshots are too correlated  
  - Ensemble diversity is insufficient  

- **Seed ensembling is not suitable here:**
  - High variance across runs   
  - Ensemble averaging hurts extreme predictions  
