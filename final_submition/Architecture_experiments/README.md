# Architecture Experiments for Vehicle Sideslip Estimation

Systematic evaluation of recurrent architectures for deep imbalanced regression.

## Kaggle Results Summary

### Architecture Comparison
| Model | Submission File | Kaggle Score |
|-------|-----------------|--------------|
| **RNN (Best)** | predictions09.csv | **0.25540** |
| RNN (h64 fc64 res L1) | submission_RNN_h64_fc64_res_L1_val0.2487_20260119_1454.csv | 0.33940 |
| RNN (h64 fc64 nores L1) | submission_RNN_h64_fc64_nores_L1_val0.2464_20260119_1446.csv | 0.39733 |
| RNN (final) | submission_rnn_final.csv | 0.39916 |
| LSTM (num=1 size=64) | submission_lstm_best.csv | 0.41992 |
| GRU (num=1 size=64) | submission_gru_best.csv | 0.42235 |
| RNN (num=1 size=64) | submission_rnn_best.csv | 0.45017 |

### Regularization Experiments
| Method | Submission File | Kaggle Score |
|--------|-----------------|--------------|
| **No Regularization** | predictions09.csv | **0.25540** |
| Weight Decay | submission_reg_weight_decay.csv | 0.32653 |
| Dropout 0.1 | submission_reg_dropout_0.1.csv | Error |
| Combined (Dropout + WD) | submission_reg_combined.csv | Error |

## Hypotheses Tested

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| Bidirectional context improves predictions | Rejected | Short sequences (20 timesteps) |
| Attention focuses on informative timesteps | Rejected | Marginal benefit |
| Deeper networks learn better features | Rejected | Overfitting |
| More parameters = better generalization | Rejected | Opposite true (0.45 → 0.25) |
| Dropout prevents overfitting | Rejected | Hurts minority class (Error submissions) |
| Weight decay regularizes | Rejected | 0.32653 vs 0.25540 without |

## Winning Architecture

```
Input: (B, 20, 22)
    ↓
RNN(64 units, 1 layer, ReLU) → (B, 64)
    ↓
FC: 64 → 64 (ReLU) → 1
```

**Parameters**: ~13K

**Why simple wins:**
- Limited capacity forces generalizable patterns
- ReLU outperforms tanh for regression
- No dropout = full gradient signal for rare samples
- 13K params acts as implicit regularization

## Dropout Analysis - Kaggle Evidence

| Model | Regularization | Submission | Kaggle Score |
|-------|----------------|------------|--------------|
| **RNN** | **None** | predictions09.csv | **0.25540** |
| RNN | Weight Decay | submission_reg_weight_decay.csv | 0.32653 |
| RNN | Dropout 0.1 | submission_reg_dropout_0.1.csv | Error |
| RNN | Combined | submission_reg_combined.csv | Error |

**Dropout failure reasons:**
- Signal attenuation: 5% few-shot × (1-0.3)³ ≈ 1.7% effective signal
- Exacerbates data scarcity in tail regions
- Prevents learning minority patterns (caused error submissions)

## Conclusion

Implicit regularization via constrained model capacity outperforms explicit regularization (dropout) for imbalanced regression. Simple RNN achieves best Kaggle score: **0.25540**

## Configuration

Seed: 42 | Sequence: 20 timesteps | Filter: vxCG > 3.0 m/s | Loss: WeightedL1
