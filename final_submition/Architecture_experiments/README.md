# Architecture Experiments for Vehicle Sideslip Estimation

Systematic evaluation of recurrent architectures for deep imbalanced regression.

## Results

| Model | Description | Params | Kaggle Score |
|-------|-------------|--------|--------------|
| **RNN** | Vanilla RNN (64 units, 1 layer, ReLU) | 13K | **0.2175** |
| LSTM | Unidirectional LSTM | 50K | 0.2213 |
| GRU | Gated Recurrent Unit | 45K | 0.2245 |
| BiLSTM + Residual | Bidirectional + residual connections | 400K | 0.2287 |
| BiLSTM + Attention | Bidirectional + multi-head attention | 600K | 0.2298 |
| BiLSTM | Standard Bidirectional LSTM | 300K | 0.2312 |

## Hypotheses Tested

| Hypothesis | Result |
|------------|--------|
| Bidirectional context improves predictions | Rejected (20 timesteps too short) |
| Attention focuses on informative timesteps | Rejected (marginal benefit) |
| Deeper networks learn better features | Rejected (overfitting) |
| More parameters = better generalization | Rejected (opposite true) |
| Dropout prevents overfitting | Rejected (hurts minority class) |

## Winning Architecture

```
Input: (B, 20, 22)
    ↓
RNN(64 units, 1 layer, ReLU) → (B, 64)
    ↓
FC: 64 → 64 (ReLU) → 1
```

**Why simple wins:**
- Limited capacity forces generalizable patterns
- ReLU outperforms tanh for regression
- No dropout = full gradient signal for rare samples
- 13K params acts as implicit regularization

## Dropout Analysis

| Model | Architecture | Dropout | Score |
|-------|--------------|---------|-------|
| LSTM | 128 units × 3 layers | [0.4, 0.3, 0.2] | ~0.60-0.80 |
| GRU | 256 units × 3 layers | [0.35, 0.25, 0.15] | ~0.40-0.45 |
| **RNN** | **64 units × 1 layer** | **None** | **0.2175** |

**Dropout failure reasons:**
- Signal attenuation: 5% few-shot × (1-0.3)³ ≈ 1.7% effective signal
- Exacerbates data scarcity in tail regions
- Prevents learning minority patterns

## Conclusion

Implicit regularization via constrained model capacity outperforms explicit regularization (dropout) for imbalanced regression. Simple architectures generalize better when minority class learning is critical.

## Configuration

Seed: 42 | Sequence: 20 timesteps | Filter: vxCG > 3.0 m/s | Loss: WeightedL1
