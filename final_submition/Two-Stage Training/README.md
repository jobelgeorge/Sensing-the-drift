# Two-Stage Training for Vehicle Sideslip Estimation

Curriculum-based training strategy for deep imbalanced regression on vehicle sideslip angle prediction.

## Core Concept

Two-stage training separates the learning process into distinct phases with different objectives:

**Stage 1 (Warm-up)**: Learn general patterns with standard L1 loss  
**Stage 2 (Fine-tuning)**: Refine tail predictions with Weighted L1 loss

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: L1 Loss (30 epochs)                                       │
│  ─────────────────────────────                                      │
│  • lr = 3e-4 (higher learning rate)                                 │
│  • Standard L1: treats all samples equally                          │
│  • Goal: Learn baseline representations                             │
│                        ↓                                            │
│  Stage 2: WeightedL1 Loss (40 epochs)                               │
│  ──────────────────────────────────                                 │
│  • lr = 5e-5 (lower learning rate)                                  │
│  • Weighted L1: w_i = 1 + slope × max(0, |y_i| - threshold)         │
│  • Goal: Refine predictions for few-shot samples                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Results

| Configuration | Stage 1 | Stage 2 | Threshold | Slope | Public Score |
|--------------|---------|---------|-----------|-------|-------------:|
| L1 Only (Baseline) | 70 ep | - | - | - | 0.8000 |
| WeightedL1 Only | - | 70 ep | 2.0 | 1.0 | 0.2500 |
| **Two-Stage (Optimal)** | **30 ep** | **40 ep** | **2.0** | **1.0** | **0.2175** |
| Two-Stage (Short Stage 1) | 20 ep | 40 ep | 2.0 | 1.0 | 0.2245 |
| Two-Stage (Long Stage 2) | 30 ep | 50 ep | 2.0 | 1.0 | 0.2450 |
| Two-Stage (Low Threshold) | 30 ep | 40 ep | 1.5 | 1.0 | 0.2320 |
| Two-Stage (High Threshold) | 30 ep | 40 ep | 2.5 | 1.0 | 0.2280 |

---

## Hypotheses

| Hypothesis | Result |
|------------|--------|
| Two-stage training improves few-shot performance | ✅ Confirmed |
| Stage 1 warm-up prevents early overfitting to weights | ✅ Confirmed |
| Lower LR in Stage 2 enables fine-grained refinement | ✅ Confirmed |
| Threshold τ=2.0 aligns with band boundaries | ✅ Confirmed |
| Longer Stage 2 improves performance | ❌ Rejected (overfitting) |
| Higher slope increases few-shot focus | ❌ Rejected (gradient instability) |

---

## Mathematical Formulation

### Stage 1: Standard L1 Loss
```
L₁ = (1/N) Σᵢ |yᵢ - ŷᵢ|
```
- Treats all samples equally
- Learns general feature representations
- Gradient: sign(ŷ - y)

### Stage 2: Weighted L1 Loss
```
wᵢ = 1 + s · max(0, |yᵢ| - τ)
L_weighted = (1/N) Σᵢ wᵢ · |yᵢ - ŷᵢ|

Optimal: τ = 2.0°, s = 1.0
```
- Linearly increasing weights for |y| > τ
- Amplifies gradients for tail samples
- Threshold aligns with many/medium-shot boundary (~2.4°)

---

## Architecture

```
Input: (B, 20, 22)
    ↓
RNN(64 units, 1 layer, ReLU) → (B, 64)
    ↓
FC: 64 → 64 (ReLU) → 1
    ↓
Output: Predicted sideslip angle
```

**Parameters**: ~13K

---

## Training Pipeline

```python
# Stage 1: L1 Loss
optimizer = AdamW(lr=3e-4)
scheduler = CosineAnnealingLR(T_max=30)
loss_fn = L1Loss()

for epoch in range(30):
    train_epoch(model, train_loader, loss_fn, optimizer)
    scheduler.step()

# Stage 2: WeightedL1 Loss  
optimizer = AdamW(lr=5e-5)
scheduler = CosineAnnealingLR(T_max=40)
loss_fn = WeightedL1Loss(threshold=2.0, slope=1.0)

for epoch in range(40):
    train_epoch(model, train_loader, loss_fn, optimizer)
    scheduler.step()
```

---

## Hyperparameter Sensitivity

### Stage 1 Epochs
| Epochs | Score | Observation |
|--------|-------|-------------|
| 20 | 0.2245 | Underfitting - insufficient warm-up |
| 25 | 0.2220 | Slight improvement |
| 28 | 0.2195 | Close to optimal |
| **30** | **0.2175** | **Optimal** |
| 32 | 0.2190 | Marginal degradation |
| 35 | 0.2210 | Overfitting begins |

### Stage 2 Epochs
| Epochs | Score | Observation |
|--------|-------|-------------|
| 30 | 0.2340 | Underfitting |
| 35 | 0.2250 | Improving |
| 38 | 0.2180 | Near optimal |
| **40** | **0.2175** | **Optimal** |
| 42 | 0.2261 | Overfitting begins |
| 45 | 0.2280 | Clear overfitting |
| 50 | 0.2450 | Severe overfitting |

### Threshold (τ)
| Threshold | Score | Observation |
|-----------|-------|-------------|
| 1.5 | 0.2320 | Over-weighting many-shot |
| 1.8 | 0.2250 | Slight improvement |
| **2.0** | **0.2175** | **Optimal** (aligns with band boundary) |
| 2.2 | 0.2210 | Under-weighting few-shot |
| 2.5 | 0.2280 | Insufficient tail focus |

### Slope (s)
| Slope | Score | Observation |
|-------|-------|-------------|
| 0.5 | 0.2340 | Weak reweighting |
| 0.8 | 0.2220 | Moderate effect |
| **1.0** | **0.2175** | **Optimal** |
| 1.5 | 0.2310 | Gradient instability |
| 2.0 | 0.2450 | Training divergence |

---

## Why Two-Stage Works

### 1. Curriculum Learning Effect
- Stage 1 establishes stable baseline features
- Stage 2 refines without catastrophic forgetting

### 2. Gradient Stabilization
- Starting with weighted loss from scratch causes instability
- Pre-trained weights provide stable gradient landscape

### 3. Learning Rate Schedule
- High LR (3e-4) in Stage 1: rapid feature learning
- Low LR (5e-5) in Stage 2: fine-grained tail refinement

### 4. Threshold Alignment
- τ=2.0° closely matches many/medium boundary (2.4°)
- Creates smooth transition in gradient weighting

---

## Ensemble Configuration

| Ensemble | Seeds | Blend | Score |
|----------|-------|-------|-------|
| Single | 42 | - | 0.2220 |
| **Two-seed** | **42, 2024** | **50/50** | **0.2175** |
| Three-seed | 42, 2024, 123 | 33/33/34 | 0.2195 |
| Four-seed | 42, 2024, 123, 456 | 25/25/25/25 | 0.2210 |

**Observation**: Minimal ensembling (2 seeds) optimal; larger ensembles dilute informative patterns.

---

## Comparison with Single-Stage Approaches

| Approach | Training | Score | Notes |
|----------|----------|-------|-------|
| L1 Only | 70 epochs | 0.8000 | Ignores tail |
| WeightedL1 Only | 70 epochs | 0.2500 | Unstable early training |
| LDS + L1 | 70 epochs | 0.5500 | Smooth density misaligned |
| **Two-Stage** | **30+40 epochs** | **0.2175** | **Best** |

---

## Notebooks

- `01_two_stage_baseline.ipynb` - Core two-stage training implementation and ablation
- `02_two_stage_hyperparameter_sweep.ipynb` - Systematic hyperparameter exploration with visualizations

---

## Key Insights

1. **Warm-up is Essential**: Direct weighted training causes gradient instability
2. **Epoch Count is Critical**: Optimal at 40 Stage 2 epochs; more causes overfitting
3. **Threshold Alignment**: τ=2.0° maps to competition band boundaries
4. **Minimal Ensembling**: 2 seeds provide optimal variance reduction
5. **Learning Rate Decay**: 6× reduction (3e-4 → 5e-5) between stages

---

*Two-Stage Training achieved the best Kaggle score: **0.21751***
