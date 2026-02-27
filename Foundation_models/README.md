# Foundation Models for Vehicle Sideslip Estimation

Pre-trained time series foundation models evaluated for deep imbalanced regression.

## Results Summary

| Model | Few-Shot MAE | Status |
|-------|:------------:|--------|
| **RNN (Best)** | **0.2175** | Baseline |
| Chronos Base (Fine-tuned) | 0.5210 | Passed threshold, 2.4x worse than RNN |
| Chronos Small (Fine-tuned) | 0.5430 | Passed threshold, 2.5x worse than RNN |
| TimePFN | 0.5500 | Passed threshold, 2.5x worse than RNN |
| Chronos Zero-Shot | ~0.60 | Failed (many-shot MAE=0.295 > 0.2 threshold) |

Only Chronos Zero-Shot failed the many-shot MAE <= 0.2 constraint. The fine-tuned models passed the threshold but still produced 2.4-2.5x higher few-shot MAE than the simple RNN.

## Hypotheses

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| Foundation model patterns transfer | Rejected | All few-shot MAE > 0.52 |
| Zero-shot inference works | Rejected | Chronos zero-shot many-shot MAE=0.295, failed threshold |
| Fine-tuning improves foundation models | Rejected | Chronos Base fine-tuned: 0.5210, still 2.4x worse than RNN |
| Larger models (200M) improve results | Rejected | Chronos Base (200M): 0.5210 vs Small: 0.5430, marginal gain |

## Chronos Experiments

T5-based encoder-decoder, tokenizes time series as language tokens.

| Configuration | Few-Shot MAE | Notes |
|---------------|:------------:|-------|
| Zero-Shot + Ridge | ~0.60 | Many-shot MAE=0.295, failed threshold |
| Small (Fine-tuned) | 0.5430 | Passed threshold |
| Base 200M (Fine-tuned) | 0.5210 | Passed threshold, best foundation model |

**Failure Analysis:** Forecasting model architecture misaligned with state estimation task.

## TimePFN Experiments

Transformer-based prior-fitted network for in-context learning.

| Configuration | Few-Shot MAE | Notes |
|---------------|:------------:|-------|
| TimePFN | 0.5500 | Passed threshold |

**Failure Analysis:** Sequence length mismatch (96 expected, 20 available).

## Why Foundation Models Underperformed

1. **Task Mismatch**: Foundation models designed for forecasting, not state estimation
2. **Sequence Length**: TimePFN requires 96 timesteps, data has 20
3. **Domain Gap**: Vehicle dynamics patterns not in pre-training data
4. **Chronos Zero-Shot**: Only model that failed the many-shot MAE <= 0.2 constraint

## Conclusion

Task-specific models with appropriate loss functions (RNN + two-stage training) outperform foundation models for domain-specific imbalanced regression.

**Best Kaggle Score (Simple RNN):** 0.25540

## Configuration

Seed: 42 | Train/Val: 85/15 | Sequence: 20 timesteps | Filter: vxCG > 3.0 m/s
