# Sideslip Angle Prediction - Winning Solution

**Kaggle Score: 0.21699**

## Winning Configuration

| Component | Setting |
|-----------|---------|
| Model | Vanilla RNN (64 units, 1 layer, ReLU) |
| Parameters | ~13K |
| Stage 1 | L1 Loss, 32 epochs, lr=3e-4 |
| Stage 2 | WeightedL1, 40 epochs, lr=5e-5 |
| Ensemble | Seeds [42, 2024], 50/50 blend |
| Data | batch=64, seq_len=20, vxCG > 3.0 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run winning solution
cd final_submition/Winning_Solution
python winning_solution.ipynb
```

## Project Structure

```
├── Winning_Solution/              # Final winning solution
├── Architecture_experiments/      # Model architecture experiments
├── Loss_functions/                # Loss function experiments
├── Two-Stage Training/            # Two-stage training experiments
├── Ensemble_Approach/             # Ensemble experiments
├── Foundation_models/             # Chronos, TimePFN experiments
├── VAE_experiments/               # VAE experiments
├── models/                        # Model definitions
├── utils/                         # Utilities
└── datasets_challenge/            # Competition data
```

## Key Findings

- Simple RNN beats complex architectures (LSTM, BiLSTM, Transformer)
- Two-stage training: L1 → WeightedL1 for few-shot emphasis
- 2 seeds optimal; more dilutes patterns
- No regularization (dropout hurt performance)

## Team

**MAIT Group 2** - TH Köln

## Competition

[Kaggle: Sensing the Drift](https://www.kaggle.com/t/c6b20902477e482c94400773be035fa2) (Private)
