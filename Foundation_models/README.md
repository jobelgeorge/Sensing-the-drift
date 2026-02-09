# Foundation Models for Vehicle Sideslip Estimation

Pre-trained time series foundation models evaluated for deep imbalanced regression.

## Kaggle Results Summary

| Model | Status | Notes |
|-------|--------|-------|
| Vanilla RNN (Baseline) | **0.25540** | predictions09.csv |
| Chronos Zero-Shot | No submission | Failed many-shot threshold |
| Chronos Fine-Tuned | No submission | Failed many-shot threshold |
| TimePFN Zero-Shot | No submission | Failed many-shot threshold |
| TimePFN Fine-Tuned | No submission | Failed many-shot threshold |

**Note:** Foundation model experiments did not produce valid Kaggle submissions as they failed the many-shot MAE ≤ 0.2 constraint.

## Comparison with Submitted Models

### Best Kaggle Submissions (for reference)
| Model | Submission File | Kaggle Score |
|-------|-----------------|--------------|
| **RNN Two-Stage** | predictions09.csv | **0.25540** |
| VAE (beta=0.005) | submission_beta_0.005.csv | 0.25319 |
| Focal L1 (2-stage) | submission_ensemble_focal_l1.csv | 0.25913 |
| L1 Only | submission_L1_val0.1551_20260207_1952.csv | 0.27279 |

## Hypotheses

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| Foundation model patterns transfer | Rejected | Failed threshold constraint |
| Zero-shot inference works | Rejected | Many-shot MAE > 0.2 |
| Fine-tuning improves foundation models | Rejected | Still failed threshold |
| Larger models (200M) improve results | Rejected | No valid submission |

## Chronos Experiments

T5-based encoder-decoder, tokenizes time series as language tokens.

| Configuration | Result |
|---------------|--------|
| Zero-Shot + Ridge | Failed many-shot threshold |
| Fine-Tuned MLP | Failed many-shot threshold |
| Chronos-Base 200M | Failed many-shot threshold |

**Failure Analysis:** Forecasting model architecture misaligned with state estimation task.

## TimePFN Experiments

Transformer-based prior-fitted network for in-context learning.

| Configuration | Result |
|---------------|--------|
| Zero-Shot | Failed many-shot threshold |
| Fine-Tuned | Failed many-shot threshold |

**Failure Analysis:** Sequence length mismatch (96 expected, 20 available).

## Why Foundation Models Failed

1. **Task Mismatch**: Foundation models designed for forecasting, not state estimation
2. **Sequence Length**: TimePFN requires 96 timesteps, data has 20
3. **Many-shot Constraint**: Must achieve MAE ≤ 0.2 on common samples first
4. **Domain Gap**: Vehicle dynamics patterns not in pre-training data

## Conclusion

Task-specific models with appropriate loss functions (RNN + two-stage training) outperform foundation models for domain-specific imbalanced regression.

**Best Kaggle Score (Simple RNN):** 0.25540

## Configuration

Seed: 42 | Train/Val: 85/15 | Sequence: 20 timesteps | Filter: vxCG > 3.0 m/s
