# Weighted Sampler Experiments for Deep Imbalanced Regression

---

## Document Structure

1. **Best Results** – Quick summary of the winning sampling strategy   
2. **Method Overview** – How `WeightedRandomSampler` works (before experiments)  
3. **Experimental Results** – Empirical validation  
4. **Analysis & Interpretation** – Why it works (and when it fails)  
5. **Implementation Guide** – Practical usage details   

---

## Best Result Summary

| Method | Public Score |
|-------|-------------:|
| Weighted Sampler | **0.25540 (Best)** |
| Weighted Sampler + Snapshot Ensemble | 0.26166 |
| Weighted Sampler + Seed Ensemble | 0.33273 |


### Configuration

- **Sampling weights**: many = 1, medium = 15, few = 50  
- **Weight exponent**: α = 0.5  
- **Sampler**: `WeightedRandomSampler(replacement=True)`  
- **Architecture**: Vanilla RNN (64 hidden units, 1 layer)
- **Input:** 20 timesteps × 22 features 

---

## Method Overview: Weighted Random Sampling

### Core Idea

Instead of changing the data or loss, we change **how often samples are seen during training**.

Each training sequence is assigned a sampling weight based on its target value:

- **Few-shot** → high weight → sampled more often  
- **Many-shot** → low weight → sampled less often  

Sampling is performed **with replacement**, allowing rare samples to appear multiple times per epoch.

---

### What Changes vs What Stays the Same

#### Changes
- Probability of selecting each sequence
- Expected gradient contribution per target region

#### Does NOT Change
- Dataset size
- Feature values
- Temporal order within each sequence
- Validation or test distributions

This is **implicit oversampling** — exposure changes, data does not.

---

## Mathematical Intuition

Let each training sequence $i$ have weight $w_i$.

The expected SGD gradient becomes:

$$
\mathbb{E}[\nabla \mathcal{L}] 
= \sum_i p(i)\,\nabla \ell_i
\quad \text{where} \quad
p(i) = \frac{w_i}{\sum_j w_j}
$$

Thus:

- Rare samples contribute **more frequently** to optimization
- No artificial duplication of data
- No modification to loss gradients

---

## Experimental Results


**Observation**  
Weighted sampling increases rare-region exposure by **6–10× per epoch**.

---

## Results Summary

| Method | Observation |
|-------|-------------|
| Weighted Sampler | Strong baseline, best overall performance |
| Introduction of Snapshot Ensemble | Slight degradation, limited diversity between snapshots |
| Introduction of Seed Ensemble | Significant performance drop due to instability |

---

## Relationship to Prior Work

This approach aligns conceptually with:

- **Branco et al. (2019)** – Importance weighting for imbalanced regression   

Unlike explicit oversampling, **no synthetic or duplicated data** is introduced.

---

## Implementation Guide

```python
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,
    shuffle=False
)
