# Winning Solution: Two-Stage RNN with Seed Ensemble

## Best Kaggle Score: 0.21699 (Public Leaderboard)

**Submission File**: `pred_v15_s1_32ep.csv`  
**Original Source**: `notebooks_Mohamed/tune_rnn/task15_finetune_winner.ipynb`

---

## Winning Configuration

| Component | Configuration |
|-----------|---------------|
| **Model** | Vanilla RNN |
| **Hidden Units** | 64 |
| **Layers** | 1 |
| **Activation** | ReLU (not tanh) |
| **Total Parameters** | ~13K |

### Training Strategy

| Phase | Epochs | Loss Function | Learning Rate |
|-------|--------|---------------|---------------|
| **Stage 1** | **32** | L1 Loss (MAE) | 3e-4 |
| **Stage 2** | 40 | WeightedL1 (threshold=2.0, slope=1.0) | 5e-5 |

### Ensemble Configuration

| Setting | Value |
|---------|-------|
| Seeds | [42, 2024] |
| Blend Ratio | 50% Baseline + 50% Two-Stage |
| Train Ratio | 0.98 |
| Batch Size | 64 |
| Sequence Length | 20 |

---

## Key Finding: Stage 1 = 32 Epochs

The winning configuration uses **32 epochs in Stage 1** (instead of the default 30). This seemingly small change improved the Kaggle score from 0.21751 to **0.21699**.


## Why This Solution Won

### What Worked

1. **Simple vanilla RNN** with 64 hidden units beats all complex architectures (BiLSTM, LSTM, GRU, Transformer)
2. **Two-stage training**: L1 first (32 epochs), then WeightedL1 (40 epochs)
3. **Exactly 2 seeds** [42, 2024] with 50/50 blend
4. **Train ratio 0.98** - use almost all data for training
5. **Stage 2 LR = 5e-5** - very low for fine-tuning
6. **No dropout/regularization** - regularization hurt performance

### What Didn't Work

| Approach | Result |
|----------|--------|
| Complex architectures (BiLSTM, Transformer) | Worse than simple RNN |
| More than 2 seeds | Dilutes good patterns |
| Non-50/50 blend ratios | Any deviation hurts |
| More than 40 Stage 2 epochs | Overfits (val ↓ but Kaggle ↑!) |
| Dropout, BatchNorm | Regularization hurt |
| Larger models (128, 256 units) | Overfit |

### Counter-Intuitive Findings

| Expectation | Reality |
|-------------|---------|
| Deeper = better | 1-layer RNN beats 2-3 layer |
| Wider = better | 64 hidden beats 128/256 |
| LSTM/GRU > RNN | Vanilla RNN wins |
| More epochs = better | 40 is sweet spot |
| Lower val loss = better | Anti-correlates after epoch 40! |
| More seeds = better | 2 seeds optimal |

---

## Competition Context

**Task**: Predict vehicle sideslip angle from 22 sensor measurements  
**Metric**: MAE on **few-shot** region (rare extreme values |sideSlip| > 5°)  
**Constraint**: MAE on **many-shot** region must be ≤ 0.2° (hard threshold)  
**Output**: 49,328 predictions

---
