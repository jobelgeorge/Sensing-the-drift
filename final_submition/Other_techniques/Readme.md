# Sideslip Angle Estimation with LDS and FDS

This repository contains experiments on **imbalanced regression** for vehicle sideslip angle estimation using a **Vanilla RNN** model.  
We investigate two imbalance-handling techniques:

- **LDS (Label Distribution Smoothing)**  
- **FDS (Feature Distribution Smoothing)**  

The goal is to reduce error on rare (few-shot) regions while maintaining acceptable performance on frequent (many-shot) regions.

---

## 1. Problem Overview

The target variable **sideslip angle** has a **highly imbalanced distribution**:

- Many samples clustered near zero (many-shot region)
- Very few samples at extreme values (few-shot region)

To address this, we experiment with LDS and FDS as proposed in:

> **Delving into Deep Imbalanced Regression (ICML 2021)**

---

## 2. Model Architecture

All experiments use the same base model for fair comparison.

### Vanilla RNN
- Input: 20 timesteps × 22 features
- RNN:  
  - Hidden size: 48 or 64  
  - 1 layer  
  - ReLU activation  
- Output: Single regression value (sideslip angle)

---

## 3. Dataset Splits

We evaluate robustness using **four different train–validation splits**:

| Split | Description |
|-----|------------|
| **V1** | Original random split |
| **V2** | Temporal split (time-based) |
| **V3** | Randomized split |
| **V4** | File-based split (held-out file) |

Each split is trained **independently** and evaluated on the same test set.

---

## 4. LDS: Label Distribution Smoothing

### Core Idea
LDS reweights samples based on how rare their **labels** are.

Steps:
1. Bin training labels into fixed bins
2. Count samples per bin
3. Smooth label histogram using a Gaussian kernel
4. Assign higher loss weights to rare bins
5. Train model with weighted loss

### Loss Used
- `SmoothL1Loss`
- Reweighting: `sqrt_inv`
- Bins: 100
- Kernel: Gaussian (ks=5, σ=2.0)

### Results (Public Score ↓ better)

| Split | Public Score |
|-----|------|
| LDS V1 | **0.28695** |
| LDS V2 | 0.36409 |
| LDS V3 | 0.45150 |
| LDS V4 | 0.32742 |

**Observation:**  
LDS significantly improves performance compared to FDS and baseline models, especially on splits V1 and V4.

---

## 5. FDS: Feature Distribution Smoothing

### Core Idea
FDS operates on **latent features**, not labels.

Steps:
1. Extract penultimate-layer features
2. Maintain per-bin feature mean and variance
3. Smooth feature statistics across bins
4. Calibrate features before final regression

### Key Difference from LDS
| LDS | FDS |
|---|---|
| Acts on labels | Acts on features |
| Changes loss weighting | Changes representation |
| Simpler, more stable | More sensitive to tuning |

### Results (Public Score ↓ better)

| Split | Score |
|-----|------|
| FDS V1 | 0.61806 |
| FDS V2 | 0.48196 |
| FDS V3 | 0.62249 |
| FDS V4 | 0.44314 |

**Observation:**  
FDS underperformed LDS in this task, especially in many-shot regions, likely due to over-smoothing of well-represented features.

---

## 6. Key Takeaways

- **LDS consistently outperforms FDS** for this dataset
- Reweighting labels is more effective than feature calibration here
- Many-shot error is easier to control with LDS
- FDS requires careful tuning and can degrade frequent-region accuracy

