# Foundation Models for Vehicle Sideslip Estimation

Pre-trained time series foundation models evaluated for deep imbalanced regression.

## Results

| Model | Parameters | Kaggle Score | Status |
|-------|------------|--------------|--------|
| Vanilla RNN (Baseline) | 13K | 0.217 | Best |
| Chronos Zero-Shot | t5-small + Ridge | Failed | Threshold exceeded |
| Chronos Fine-Tuned | t5-small + MLP | 0.543 | 2.5x worse |
| Chronos-Base | t5-base 200M | 0.55 | No improvement |
| TimePFN Zero-Shot | Balanced context | 0.65 | 3x worse |
| TimePFN Fine-Tuned | Custom tuning | 0.60 | 2.8x worse |

## Hypotheses

| Hypothesis | Result |
|------------|--------|
| Foundation model patterns transfer to vehicle dynamics | Rejected |
| Zero-shot inference provides reasonable estimates | Rejected |
| Fine-tuning improves foundation model performance | Partial |
| Larger models (200M vs 46M) improve results | Rejected |

## Chronos Experiments

T5-based encoder-decoder, tokenizes time series as language tokens.

| Configuration | Score |
|---------------|-------|
| Zero-Shot + Ridge | Failed threshold |
| Fine-Tuned MLP | 0.543 |
| Chronos-Base 200M | 0.55 |

Failure: Forecasting model misaligned with state estimation task.

## TimePFN Experiments

Transformer-based prior-fitted network for in-context learning.

| Configuration | Score |
|---------------|-------|
| Zero-Shot | 0.65 |
| Fine-Tuned | 0.60 |

Failure: Sequence length mismatch (96 expected, 20 available).

## Conclusion

Task-specific models with appropriate loss functions outperform foundation models for domain-specific imbalanced regression.

## Configuration

Seed: 42 | Train/Val: 80/20 | Sequence: 20 timesteps | Filter: vxCG > 3.0 m/s
