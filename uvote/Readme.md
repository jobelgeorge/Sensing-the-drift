# U-Vote (Uncertainty-Aware Voting) for Deep Imbalanced Regression

Systematic evaluation of **uncertainty-aware regression** using **U-Vote** for vehicle sideslip angle prediction under **severe data imbalance**.

This work studies how **learning predictive uncertainty**, combined with **distribution-aware smoothing (LDS/FDS)**, improves robustness and generalization in imbalanced regression settings.

---

## Best Results — Quick Summary

| Method         | Public Score |
|----------------|-------------:|
| U-Vote         | 0.31122 |
| U-Vote + LDS   | 0.31113 |
| **U-Vote + FDS** | **0.31051 (Best)** |

**Key takeaway:**  
Feature-level smoothing (FDS) provides the strongest improvement when combined with uncertainty-aware learning.

---

## Motivation  
### Problem: Imbalanced Regression with Uncertainty Blindness

The sideslip angle distribution is **heavily skewed**:

- **Many-shot region (~97%)**: Near-zero sideslip  
- **Medium-shot (~2%)**: Transitional maneuvers  
- **Few-shot (~1%)**: Extreme drift / safety-critical cases  

Standard regression models:

- Optimize for mean performance  
- Treat all prediction errors equally  
- Provide **no notion of confidence**  

This leads to:

- Overconfident predictions in ambiguous regions  
- Poor handling of rare but critical samples  
- Misleading validation metrics  

---

## Research Question

**Can uncertainty-aware learning improve robustness and generalization in deep imbalanced regression, beyond simple error minimization?**

**Sub-questions:**
- Does predicted uncertainty align with true prediction difficulty?
- Can uncertainty-aware loss reduce overconfidence?
- How do LDS and FDS interact with uncertainty modeling?

---

## Method Overview: U-Vote

U-Vote extends regression models to predict:

- **μ(x)** → mean prediction  
- **σ²(x)** → input-dependent uncertainty (heteroscedastic)

U-Vote optimizes a weighted heteroscedastic negative log-likelihood:

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

## Distribution Smoothing Extensions

### Label Distribution Smoothing (LDS)

- Smooths label histogram using a Gaussian kernel  
- Reduces sharp weight transitions between bins  
- Stabilizes training in sparse regions  

**Effect:** Minor improvement, limited by label overlap.

---

### Feature Distribution Smoothing (FDS)

- Smooths **latent features**, not labels  
- Reduces representation bias caused by imbalance  
- Preserves uncertainty learning while improving generalization  

**Effect:** Best overall performance (**0.31051**).

---

## Uncertainty Calibration Analysis

**Key empirical finding:**  
Prediction error increases exactly where predicted uncertainty increases.

**Observed behavior:**
- Low error → low σ  
- High error → high σ  
- Extreme sideslip values are **not automatically uncertain**  
- Ambiguous dynamics (e.g., braking + steering) dominate uncertainty  

This confirms **meaningful aleatoric uncertainty learning**.

---

## Hypothesis Validation

| Hypothesis | Result |
|----------|--------|
| Uncertainty aligns with error | Confirmed |
| Extreme values always uncertain |  Rejected |
| LDS stabilizes learning |  Partial |
| FDS improves generalization | Confirmed |
| Validation MAE predicts Kaggle |  Weak correlation |

---

## Implementation Guide

### Model
```python
mu, log_var = model(X)

loss = weighted_heteroscedastic_nll(mu, log_var, y, w)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

## Key Takeaways

- U-Vote enables **trust-aware regression**
- Uncertainty reflects **prediction difficulty**, not label rarity
- Feature smoothing (FDS) complements uncertainty learning best
- Validation MAE alone is insufficient for imbalanced regression