# Sensing the Drift Challenge

This repository contains the work of **MAIT Group 2** for the case study **"Sensing the Drift."**

## Project Overview

This project focuses on developing virtual sensor models to predict vehicle sideslip angle (VSA) while addressing data imbalance challenges. The goal is to create accurate VSA estimates for Advanced Driver-Assistance Systems (ADAS) using standard sensor data instead of expensive dedicated sensors.

## Sideslip Angle Prediction - Winning Solution

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

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation and Running the Best Solution

1. Clone this repository:
   ```bash
   git clone <gitlab-repo-url>
   cd MAIT_G2_Drift_Challenge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run winning solution
   ```bash
   cd Winning_Solution
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
├── uvote/                         # U-Vote experiments
├── weighted Sampler/              # Weighted Sampler Experiments
├── Residual_mapping/              # Residual Mapping Experiments
├── models/                        # Model definitions
├── utils/                         # Utilities
└── datasets_challenge/            # Competition data
```

## Key Findings

- Simple RNN beats complex architectures (LSTM, BiLSTM, Transformer)
- Two-stage training: L1 → WeightedL1 for few-shot emphasis
- 2 seeds optimal; more dilutes patterns
- No regularization (dropout hurt performance)

## Team Members

**MAIT Group 2**
- Member 1: [Mohamed Desouky]
- Member 2: [Jobel Jose George]
- Member 3: [Alex Prediger]

## Competition

[Kaggle: Sensing the Drift](https://www.kaggle.com/t/c6b20902477e482c94400773be035fa2) (Private)
