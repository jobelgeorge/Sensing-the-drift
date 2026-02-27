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

## Method Overview: U-Vote

U-Vote extends regression models to predict:

- **μ(x)** → mean prediction  
- **σ²(x)** → input-dependent uncertainty (heteroscedastic)

U-Vote optimizes a weighted heteroscedastic negative log-likelihood:

```python
    # inverse variance (confidence)
    inv_var = torch.exp(-log_var)

    # Gaussian negative log-likelihood
    nll = 0.5 * inv_var * (y - mu) ** 2 + 0.5 * log_var

    # U-Vote weighting
    return (w * nll).mean()
```

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

**Effect:** Best overall performance, minor improvement (**0.31051**).

---

## Uncertainty Calibration Analysis

**Key empirical finding:**  
Prediction error increases exactly where predicted uncertainty increases.

**Observed behavior:**
- Low error → low σ  
- High error → high σ  
- Extreme sideslip values are **not automatically uncertain**   

This confirms **meaningful uncertainty learning**.

---

## Implementation Guide

### Model
```python
mu, log_var = model(X)

loss = weighted_heteroscedastic_nll(mu, log_var, y, w)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

## Key Takeaways

- U-Vote enables uncertainty-aware regression under severe imbalance
- Uncertainty reflects **prediction difficulty**, not label rarity
- Rare but structurally consistent regimes show low uncertainty
- Feature smoothing (FDS) complements uncertainty learning best
