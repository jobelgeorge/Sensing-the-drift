# Data Loading Strategies & Distribution Analysis

**Discovery folder** for exploring data loading approaches and understanding distribution characteristics.

## Purpose
This folder investigates:
1. How different data splitting strategies affect model generalization
2. Which validation file best represents test distribution (file-based split)
3. Train vs test distribution mismatch (domain shift)
4. Optimal data loading configuration for imbalanced regression

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `01_distribution_analysis.ipynb` | Sideslip distribution, band mapping, class imbalance analysis |
| 2 | `02_dataloader_comparison.ipynb` | Compare V1, V2, V3, V4 dataloaders and their impact |
| 3 | `03_file_selection_analysis.ipynb` | Analyze which file to use for validation (file-based split) |
| 4 | `04_test_distribution_analysis.ipynb` | Test set feature distribution and domain shift analysis |

## Key Questions

### Distribution Analysis
- What is the exact many/medium/few-shot distribution?
- How does sideslip correlate with sensor features?
- Are there temporal patterns in the distribution?

### Dataloader Comparison  
- V1 (Original): Random split - train/val leakage?
- V2 (Temporal): Last 15% per band - temporal shift?
- V3 (Random Stratified): Better generalization?
- V4 (File-based): Single file validation - which file?

### File Selection
- Which file has best band coverage?
- Which file's distribution matches test set?
- Drift intensity per file analysis

### Test Distribution
- Feature distribution comparison (train vs test)
- Identifying potential domain shift
- Which features show covariate shift?

## Data Loaders Available

| Loader | File | Strategy | Notes |
|--------|------|----------|-------|
| V1 | `data_loading.py` | Random split | Original, possible leakage |
| V2 | `data_loadingV2.py` | Temporal stratified | Last 15% per band |
| V3 | `data_loadingV3_random.py` | Random stratified | Better test match? |
| V4 | `data_loadingV4_file.py` | File-based | Single file as val |
| Bins | `data_loadingbins.py` | Weighted sampling | Upsamples minority |

## Expected Insights
- Best splitting strategy for this dataset
- Optimal validation file for file-based approach
- Understanding of test distribution for better submission
