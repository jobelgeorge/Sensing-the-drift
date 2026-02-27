# Sensing the Drift -- MAIT Group 2

Predict vehicle sideslip angle (VSA) from 22 sensor measurements under severe target imbalance.
Metric: MAE on few-shot region (|sideSlip| > 5 deg), subject to many-shot MAE <= 0.2 deg.

**Best Kaggle Score: 0.21699**

---

## Reproduce the Winning Solution

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (recommended)

### Steps

```bash
git clone <gitlab-repo-url>
cd MAIT_G2_Drift_Challenge
pip install -r requirements.txt
cd Winning_Solution
jupyter notebook winning_solution.ipynb   # run all cells
```

The notebook trains the model from scratch and writes the submission CSV.

### Winning Configuration

| Component | Setting |
|-----------|---------|
| Model | Vanilla RNN (64 units, 1 layer, ReLU) |
| Parameters | ~13K |
| Stage 1 | L1 Loss, 32 epochs, lr=3e-4 |
| Stage 2 | WeightedL1 (threshold=2.0, slope=1.0), 40 epochs, lr=5e-5 |
| Ensemble | Seeds [42, 2024], 50/50 blend |
| Data | batch=64, seq_len=20, vxCG > 3.0 m/s, train_ratio=0.98 |

---

## Repository Structure

- [`Winning_Solution/`](./Winning_Solution/) -- Best submission (score 0.21699) -- **start here** ([README](./Winning_Solution/README.md))
- [`models/`](./models/) -- All model definitions (RNN, LSTM, GRU, BiLSTM, VAE, residual variants)
- [`utils/`](./utils/) -- Data loading, loss functions, evaluation, visualization
- [`datasets_challenge/`](./datasets_challenge/) -- Competition data
- `requirements.txt`

### Additional Methods (Appendices)

Each folder is a self-contained experiment with its own README documenting motivation, results, and conclusions.

- **[Architecture_experiments](./Architecture_experiments/README.md)** (Kaggle: 0.25540) -- RNN vs LSTM vs GRU vs BiLSTM, regularization. Simple RNN beats all; dropout/weight decay hurt.

- **[Loss_functions](./Loss_functions/README.md)** (Kaggle: 0.2643) -- L1, Focal L1, Weighted L1, HybridWeightedFocalL1, MSE, Huber. HybridWeightedFocalL1 (gamma=0.5) best single-loss; dual balancing synergizes.

- **[Two-Stage Training](./Two-Stage%20Training/README.md)** (Kaggle: 0.25540) -- L1 warm-up then WeightedL1 fine-tuning. 30 ep L1 then 40 ep WeightedL1; longer Stage 2 overfits.

- **[Ensemble_Approach](./Ensemble_Approach/Readme.md)** (Kaggle: 0.31709) -- Multi-seed, multi-architecture, snapshot ensembles. Most ensembles degrade test performance; snapshot is partial exception.

- **[Residual_mapping](./Residual_mapping/README.md)** (Kaggle: 0.2587) -- Input, hidden-state, deep residual connections. HiddenResidual reduces generalization gap by 40%.

- **[Foundation_models](./Foundation_models/README.md)** (Few-Shot MAE: 0.5210 best) -- Chronos, TimePFN (zero-shot and fine-tuned). Only Chronos Zero-Shot failed many-shot threshold. Fine-tuned models passed but 2.4x worse than RNN (0.2175).

- **[VAE_experiments](./VAE_experiments/README.md)** (Kaggle: 0.25319) -- VAE architecture/loss sweeps, latent augmentation. Underperforms baseline RNN; reconstruction/contrastive/augmentation all hurt.

- **[uvote](./uvote/Readme.md)** (Kaggle: 0.31051) -- U-Vote uncertainty-aware regression + LDS/FDS. Uncertainty aligns with error not label rarity; FDS helps most.

- **[Weighted_sampler](./weighted%20sampler/Readme.md)** (Kaggle: 0.25540) -- WeightedRandomSampler for minority oversampling. Improves rare-region exposure; ensembles on top degrade the few shot score.

- **[Data_loading_strategies](./Data_loading_strategies/README.md)** -- Distribution analysis, dataloader comparison, file-based splits. Exploratory analysis of train/test shift and splitting strategies.

- **[Other_techniques](./Other_techniques/Readme.md)** (Kaggle: 0.28695) -- LDS, FDS, RankSim (from DIR literature). LDS most effective; FDS over-smooths; RankSim strong alternative.

---

## Key Findings

1. **Simple models generalize better.** A 13K-parameter vanilla RNN outperformed LSTM, BiLSTM, GRU, Transformers, and VAEs. Limited capacity acts as implicit regularization on this small dataset.

2. **Two-stage training is critical.** L1 loss first learns general representations; WeightedL1 loss second refines tail predictions. Starting directly with weighted loss causes gradient instability.

3. **Regularization hurts.** Dropout and weight decay both degraded performance. With only ~1% few-shot samples, regularization attenuates the already-weak gradient signal from rare cases.

4. **Validation MAE is unreliable.** Weak correlation (r~0.42) between validation MAE and Kaggle score. Models with the best validation often had the worst test generalization. Monitoring the generalization gap (test - val) is essential.

5. **Ensemble diminishing returns.** Two seeds with 50/50 blend is optimal. More seeds dilute learned patterns. Multi-architecture ensembles overfit to validation distribution.

6. **Loss function design matters but architecture selection matters more.** HybridWeightedFocalL1 improved single-stage results by 3.1%, but two-stage training with simple L1/WeightedL1 on the same architecture achieved the best overall score.

7. **Foundation models fail on domain-specific tasks.** Chronos and TimePFN could not meet the many-shot constraint due to task mismatch (forecasting vs state estimation) and sequence length incompatibility.

8. **Data augmentation consistently hurts.** VAE latent augmentation, mixup, and soft augmentation all degraded performance, likely because synthesized samples do not match the true data manifold.

---

## Team

| Member |
|--------|
| Mohamed Desouky |
| Jobel Jose George |
| Alex Prediger |

## Competition

[Kaggle: Sensing the Drift](https://www.kaggle.com/t/c6b20902477e482c94400773be035fa2) (Private)
