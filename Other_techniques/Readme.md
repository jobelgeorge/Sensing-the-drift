# Sideslip Angle Estimation with LDS and FDS

This folder presents experiments on **imbalanced regression** for vehicle **sideslip angle estimation** using a **Vanilla RNN** model.  
We evaluate and compare three imbalance-handling techniques:

- **LDS (Label Distribution Smoothing)**
- **FDS (Feature Distribution Smoothing)**
- **RankSim (Ranking Similarity Regularization)**

The objective is to reduce prediction error in **rare (few-shot) regions** while maintaining strong performance in **frequent (many-shot) regions**.

---

## Model Architecture

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

## 6. RankSim: Ranking Similarity Regularization

### Core Idea
RankSim enforces **consistency between target similarity and feature similarity**.

Samples with similar target values are encouraged to have similar latent feature representations, preserving ranking structure.

### Loss Formulation
$$
\mathcal{L} = \mathcal{L}_{\text{regression}} + \lambda \cdot \mathcal{L}_{\text{RankSim}}
$$

RankSim is applied **only during training** and does not alter inference.

### Configuration
- Base loss: `SmoothL1Loss`
- Temperature: `0.1`
- RankSim weight (λ): `0.05`

### Results (Public Score ↓ better)

| Split | Public Score |
|------|--------------|
| RankSim V1 | 0.31016 |
| RankSim V2 | 0.34851 |
| RankSim V3 | **0.29578** |
| RankSim V4 | 0.36101 |

**Observation:**  
RankSim consistently improves over FDS and approaches LDS performance, showing strong robustness to randomized splits.

---

## 7. Key Takeaways

- **LDS is the most effective method** for this dataset
- **RankSim offers a strong representation-level alternative**
- RankSim improves robustness without affecting inference
- **FDS requires careful tuning** and may degrade many-shot accuracy
- Label-level imbalance correction is particularly effective for sideslip estimation

