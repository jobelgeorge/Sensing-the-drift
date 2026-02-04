# U-Vote for Vehicle Sideslip Estimation

Uncertainty-aware imbalanced regression for vehicle sideslip angle estimation using sequential vehicle dynamics data.

This project evaluates **U-Vote**, **U-Vote with Label Distribution Smoothing (LDS)**, and **U-Vote with Feature Distribution Smoothing (FDS)** on a highly imbalanced dataset.

---

## Models

### U-Vote
Heteroscedastic regression predicting mean (**μ**) and uncertainty (**σ²**) with band-aware weighting.

### U-Vote + LDS
Applies label distribution smoothing to reduce sharp frequency transitions between bands.

### U-Vote + FDS
Applies feature distribution smoothing to stabilize representations for rare sideslip regimes.

---

## Architecture

- **Backbone:** Vanilla RNN (ReLU)  
- **Input:** 20 timesteps × 22 features  
- **Output:**
  - **μ** (predicted sideslip)
  - **log σ²** (uncertainty)
- **Loss:** Weighted heteroscedastic Gaussian NLL (U-Vote loss)

---

## Submissions & Scores

| Method         | Public Score |
|----------------|-------------:|
| U-Vote         | 0.31122 |
| U-Vote + LDS   | 0.31113 |
| U-Vote + FDS   | **0.31051 (Best)** |

---

## Results Summary

| Method         | Observation |
|---------------|-------------|
| U-Vote         | Strong and stable baseline |
| U-Vote + LDS   | Not much of an improvement, sometimes degrades |
| U-Vote + FDS   | Small but consistent gain |

---

## Key Findings

- **U-Vote** significantly improves robustness over deterministic regression by:
  - Learning uncertainty
  - Preventing overconfidence in rare regimes

- **LDS** is not beneficial for this dataset:
  - Labels are already clean
  - Smoothing blurs band boundaries

- **FDS** provides mild regularization for rare features without harming many-shot regions.
