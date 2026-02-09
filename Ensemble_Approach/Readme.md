# Ensemble Learning for Imbalanced Time-Series Regression: An Empirical Study

**A Scientific Investigation of Ensemble Strategies for Vehicle Sideslip Angle Estimation**

## Abstract

This work investigates two ensemble learning strategies—**multi-seed ensembles** and **multi-architecture ensembles**—for improving vehicle sideslip angle estimation under highly imbalanced regression conditions. While ensemble methods theoretically reduce prediction variance through aggregation of diverse models, we present empirical evidence that not all ensemble strategies generalize beyond validation sets. Our experiments reveal a critical gap between validation performance and test set generalization, demonstrating that ensemble approaches can overfit to validation distributions despite strong theoretical foundations. We compare ensemble performance against simple baseline models and analyze the conditions under which ensemble methods fail, providing insights for practitioners working with drift detection in autonomous vehicle systems.

## 1. Introduction

### 1.1 Motivation

Ensemble learning has become a cornerstone technique in modern machine learning, with methods like Random Forests, Gradient Boosting, and model averaging achieving state-of-the-art results across diverse domains. The fundamental principle—combining predictions from multiple models often outperforms any single model—is supported by both theoretical analysis (bias-variance decomposition) and empirical success in competitions and production systems.

However, ensemble methods are not universally superior. Their effectiveness depends on:
1. **Model diversity**: Models must make complementary errors
2. **Individual quality**: Weak individual models produce weak ensembles
3. **Distribution alignment**: Training/validation/test distributions must be consistent
4. **Task characteristics**: Some problems benefit more from ensembles than others

For **imbalanced regression** tasks—where target distributions are highly skewed and underrepresented regions (few-shot samples) are critical—ensemble methods promise particular benefits through variance reduction and improved coverage of rare regions.

### 1.2 Research Questions

This empirical study addresses three key questions:

**RQ1**: Do multi-seed ensembles (same architecture, different initializations) improve generalization over single baseline models for imbalanced time-series regression?

**RQ2**: Do multi-architecture ensembles (diverse architectures, same seed) provide superior performance compared to multi-seed approaches?

**RQ3**: Under what conditions do ensemble methods fail to generalize from validation to test sets?

## 2. Ensemble Strategies Investigated

### 2.1 Multi-Seed Ensemble (Homogeneous)

**Theoretical Foundation**: Random initialization introduces variance in model training. By training K identical architectures with different seeds and averaging predictions, we maintain low bias (same architecture) while reducing variance (averaging K independent predictions).

**Hypothesis**: The multi-seed ensemble will outperform a single baseline model by σ/√K variance reduction, where σ is single-model variance and K is ensemble size.

**Implementation**:
- Architecture: VanillaRNN (hidden_size=64, num_layers=1)
- Ensemble size: K=5 models
- Seeds: [42, 142, 242, 342, 442]
- Aggregation: Arithmetic mean
- Training: 50 epochs, MAE loss, AdamW optimizer (lr=3e-4)

**Expected Outcome**: 1-3% MAE improvement over baseline through variance reduction.

### 2.2 Multi-Architecture Ensemble (Heterogeneous)

**Theoretical Foundation**: Architectural diversity creates models with different inductive biases, leading to complementary error profiles. When architectures capture different aspects of the data, their averaged predictions can exceed any individual model's performance.

**Hypothesis**: Multi-architecture ensembles will provide greater diversity than multi-seed ensembles, resulting in superior generalization through complementary error cancellation.

**Implementation**:
- Architectures: {VanillaRNN, SimpleLSTM, SimpleGRU, SimpleBiLSTM, BiLSTMAttention}
- Ensemble size: N=5 distinct architectures
- Single seed: 42 (isolates architectural diversity)
- Aggregation: Arithmetic mean
- Training: Identical hyperparameters across all architectures

**Expected Outcome**: 2-5% MAE improvement over best individual architecture through architectural diversity.

### 2.3 Baseline Model

**Purpose**: Establish performance floor and validate whether ensemble complexity is justified.

**Configuration**:
- Architecture: VanillaRNN (simplest architecture)
- Single seed: 42
- Training: Same hyperparameters as ensemble models

## 3. Experimental Setup

### 3.1 Dataset Characteristics

- **Task**: Vehicle sideslip angle estimation (continuous regression)
- **Input**: 20-timestep sequences of vehicle dynamics features
- **Distribution**: Highly imbalanced (most samples near zero sideslip)
- **Challenge**: Accurate prediction of rare high-sideslip events (few-shot regime)

### 3.2 Evaluation Protocol

**Validation Set** (20% holdout):
- Used for model selection and hyperparameter tuning
- Band-specific metrics: few-shot, medium-shot, many-shot MAE

**Test Set** (Kaggle competition):
- Blind evaluation on unseen data
- Ground truth not disclosed

### 3.3 Training Configuration

```python
BATCH_SIZE = 256
SEQUENCE_LENGTH = 20
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
LOSS_FUNCTION = nn.L1Loss()  # MAE
OPTIMIZER = torch.optim.AdamW
SCHEDULER = CosineAnnealingLR
```

## 4. Results

### 4.1 Validation Performance Summary

| Model Configuration | Validation MAE | Improvement vs Baseline |
|---------------------|----------------|-------------------------|
| **Baseline VanillaRNN** (seed 42) | **0.1578** | — |
| Multi-Seed Ensemble (K=5, VanillaRNN) | **0.1487** | **+5.77%** |
| Best Individual (SimpleLSTM) | **0.1464** | **+7.22%** |
| Multi-Architecture Ensemble (N=5) | **0.1416** | **+10.27%** |

**Validation Observations**:
- Both ensemble strategies outperformed the baseline
- Multi-architecture ensemble showed best validation performance
- SimpleLSTM emerged as strongest individual architecture
- All models improved upon baseline, supporting ensemble hypotheses

### 4.2 Test Performance Summary (Kaggle Leaderboard)

| Model Configuration | Test MAE | Rank vs Baseline |
|---------------------|----------|------------------|
| **Baseline VanillaRNN** (seed 42) | **0.32509** | **Best Performance** |
| Best Individual (SimpleLSTM) | **0.32627** | Inferior (+0.36%) |
| Multi-Seed Ensemble (K=5, VanillaRNN) | **0.33358** | Inferior (+2.61%) |
| Multi-Architecture Ensemble (N=5) | **0.34976** | Inferior (+7.59%) |

**Critical Finding**: All ensemble methods and the best individual model performed worse than the simple baseline on the test set.

### 4.3 Performance Gap Analysis

| Model | Val MAE | Test MAE | Generalization Gap |
|-------|---------|----------|-------------------|
| Baseline VanillaRNN | 0.1578 | 0.32509 | **+106.0%** |
| SimpleLSTM | 0.1464 | 0.32627 | **+122.9%** |
| Multi-Seed Ensemble | 0.1487 | 0.33358 | **+124.3%** |
| Multi-Architecture Ensemble | 0.1416 | 0.34976 | **+147.0%** |

**Key Observation**: More complex models showed larger generalization gaps, with the multi-architecture ensemble exhibiting the worst test performance despite best validation results.

## 5. Analysis and Discussion

### 5.1 Why Ensembles Failed

#### 5.1.1 Overfitting to Validation Distribution

The consistent pattern of validation improvement but test degradation suggests **ensemble overfitting**:

1. **Insufficient Diversity**: Multi-seed ensembles showed high prediction correlation (>0.95), indicating that different seeds produced nearly identical models. Without true diversity, averaging provides minimal benefit while increasing model complexity.

2. **Validation Set Memorization**: The multi-architecture ensemble achieved lowest validation MAE (0.1416) by fitting the validation set's specific characteristics. This "over-tuning" to validation distribution harmed test generalization.

3. **Architectural Complexity Bias**: More complex architectures (BiLSTMAttention) and ensemble aggregation may have captured validation-specific patterns that didn't transfer to test data.

#### 5.1.2 The Simplicity Advantage

**Baseline VanillaRNN succeeded because**:
- Minimal capacity reduces overfitting risk
- Simple architecture generalizes better under distribution shift
- Fewer parameters mean less validation-specific fitting
- Single model avoids ensemble aggregation errors

### 5.2 Implications for Ensemble Learning

This empirical study reveals important cautionary lessons:

#### 5.2.1 Ensemble Diversity is Critical

**Multi-seed ensembles failed** because seed variation alone doesn't create sufficient diversity. High correlation (>0.95) means averaged predictions barely differ from individual predictions.

**Multi-architecture ensembles failed worse** despite lower correlation (~0.92), suggesting that architectural diversity alone is insufficient without proper validation-test alignment.

#### 5.2.2 When to Use Ensembles

Ensembles are most effective when:
- Individual models are strong but make uncorrelated errors
- Training/validation/test distributions are well-aligned
- Sufficient training data exists to prevent overfitting
- Task benefits from variance reduction (stable predictions)

Ensembles pose higher risk when:
- Distribution shifts exist between validation and test
- Individual models already overfit validation set
- Baseline models are very simple and generalize well
- Model diversity is low (high prediction correlation)

### 5.3 Architectural Diversity Analysis

Correlation matrix from multi-architecture ensemble revealed:

```
                VanillaRNN  SimpleLSTM  SimpleGRU  SimpleBiLSTM  BiLSTMAttention
VanillaRNN         1.000      0.943      0.938       0.929          0.915
SimpleLSTM         0.943      1.000      0.971       0.965          0.948
SimpleGRU          0.938      0.971      1.000       0.968          0.951
SimpleBiLSTM       0.929      0.965      0.968       1.000          0.972
BiLSTMAttention    0.915      0.948      0.951       0.972          1.000

Mean correlation: 0.952 (High - insufficient diversity)
```

Despite using 5 different architectures, predictions remained highly correlated (>0.91), explaining why architectural diversity failed to improve test performance.

## 6. Conclusions

### 6.1 Key Findings

1. **Negative Result on Ensembles**: Both multi-seed and multi-architecture ensembles underperformed the simple baseline on test data, despite superior validation performance.

2. **Simplicity Wins**: The simplest model (VanillaRNN baseline) achieved best test performance with smallest generalization gap.

3. **Diversity is Necessary but Not Sufficient**: Architectural diversity (correlation 0.952) exceeded seed diversity (>0.95), but neither provided enough complementary errors to justify ensemble complexity.

4. **Validation Overfitting is Real**: Progressive validation improvement (baseline → single-arch → multi-seed → multi-arch) corresponded to progressive test degradation, indicating ensemble-level overfitting.

## 7. Reproducibility

### 7.1 Notebooks

- **`multi_seed_ensemble.ipynb`**: Complete implementation of K=5 VanillaRNN ensemble
- **`multi_architecture_ensemble.ipynb`**: Full multi-architecture ensemble with 5 diverse RNN architectures

### 7.2 Configuration

All experiments use identical training configurations (Section 3.3) with deterministic seeding for reproducibility.

### 7.3 Artifacts

```
Ensemble_Approach/
├── multi_seed_ensemble.ipynb           # Multi-seed implementation
├── multi_architecture_ensemble.ipynb   # Multi-architecture implementation
```