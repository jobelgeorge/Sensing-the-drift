"""
Data Loading V3: RANDOM Stratified Split (Better Test Generalization)
======================================================================
Hypothesis: Temporal split causes train/val/test mismatch.
Solution: Random stratified split by band for better distribution matching.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import glob
from pathlib import Path

try:
    from utils.band_mapping import load_band_mapping
    from utils.custom_datasets import sequential_dataset, unsupervised_sequential_dataset
except ModuleNotFoundError:
    from band_mapping import load_band_mapping
    from custom_datasets import sequential_dataset, unsupervised_sequential_dataset


def apply_threshold_v1_logic(df, threshold_column="vxCG", threshold_value=3.0, seq_length=20):
    """Apply threshold using V1 pipeline logic: group consecutive sequences."""
    df_mask = df[threshold_column] > threshold_value
    groups = df[df_mask].groupby((~df_mask).cumsum())
    
    valid_groups = []
    for _, group in groups:
        if len(group) > seq_length:
            valid_groups.append(group.reset_index(drop=True))
    
    return valid_groups


def random_stratified_split_groups(groups_list, band_dict, val_ratio=0.2, seed=42):
    """
    RANDOM split of groups by band (not temporal).
    
    Key difference from V2:
    - V2: Last X% of each band → validation (temporal)
    - V3: Random X% of each band → validation (better generalization)
    """
    np.random.seed(seed)
    
    if len(groups_list) == 0:
        return [], []
    
    # Classify each group by dominant band
    group_bands = []
    for group_df in groups_list:
        bands = group_df['sideSlip'].apply(
            lambda x: band_dict.get(round(float(x), 1), 'unknown')
        )
        dominant_band = bands.mode()[0] if len(bands) > 0 else 'unknown'
        group_bands.append(dominant_band)
    
    # Organize by band
    band_groups = {'many': [], 'medium': [], 'few': []}
    for idx, (group_df, band) in enumerate(zip(groups_list, group_bands)):
        if band in band_groups:
            band_groups[band].append((idx, group_df))
    
    train_groups = []
    val_groups = []
    
    # Random split per band
    for band_name in ['many', 'medium', 'few']:
        band_group_list = band_groups[band_name]
        
        if len(band_group_list) == 0:
            continue
        
        # SHUFFLE before split (key difference!)
        indices = np.arange(len(band_group_list))
        np.random.shuffle(indices)
        
        split_idx = int(len(band_group_list) * (1 - val_ratio))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_groups.extend([band_group_list[i][1] for i in train_indices])
        val_groups.extend([band_group_list[i][1] for i in val_indices])
    
    return train_groups, val_groups


def create_concat_dataset_from_groups(groups_list, target, seq_length, columns_to_drop):
    """Create ConcatDataset from groups."""
    datasets = []
    for group_df in groups_list:
        ds = sequential_dataset(
            df=group_df,
            target=target,
            seq_length=seq_length,
            columns_to_drop=columns_to_drop
        )
        if len(ds) > 0:
            datasets.append(ds)
    
    if len(datasets) == 0:
        empty_df = pd.DataFrame()
        return sequential_dataset(empty_df, target, seq_length, columns_to_drop)
    
    concat_ds = ConcatDataset(datasets)
    if len(datasets) > 0:
        concat_ds.X_columns = datasets[0].X_columns
    return concat_ds


def create_unsupervised_concat_dataset_from_groups(groups_list, seq_length, columns_to_drop):
    """Create ConcatDataset from groups for test data."""
    datasets = []
    for group_df in groups_list:
        ds = unsupervised_sequential_dataset(
            df=group_df,
            seq_length=seq_length,
            columns_to_drop=columns_to_drop
        )
        if len(ds) > 0:
            datasets.append(ds)
    
    if len(datasets) == 0:
        empty_df = pd.DataFrame()
        return unsupervised_sequential_dataset(empty_df, seq_length, columns_to_drop)
    
    return ConcatDataset(datasets)


def load_data(
    train_path=r"datasets_challenge\train",
    val_ratio=0.2,
    columns_to_drop=['sideSlip', 'vyCG', 'longitude', 'latitude', 'time'],
    sequence_length=20,
    target="sideSlip",
    threshold=True,
    threshold_value=3.0,
    threshold_column="vxCG",
    batchsize=64,
    scaling=True,
    seed=42,
    shuffle=True
):
    """
    Load data with RANDOM stratified split (not temporal).
    
    Critical change: Random sampling of groups per band instead of temporal split.
    Hypothesis: This will better match test set distribution.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    band_dict = load_band_mapping()
    
    print("="*80)
    print("V3 Data Loader: RANDOM Stratified Split (Better Generalization)")
    print("="*80)
    
    train_files = sorted(glob.glob(f"{train_path}/train_part_*.csv"))
    print(f"\nProcessing {len(train_files)} files...")
    
    all_groups = []
    for file_path in train_files:
        df = pd.read_csv(file_path)
        groups = apply_threshold_v1_logic(df, threshold_column, threshold_value, sequence_length)
        all_groups.extend(groups)
    
    print(f"Total groups: {len(all_groups)}")
    
    # RANDOM split (not temporal!)
    train_groups, val_groups = random_stratified_split_groups(all_groups, band_dict, val_ratio, seed)
    
    # Concat → scale → separate
    train_concat = pd.concat(train_groups, keys=range(len(train_groups)))
    val_concat = pd.concat(val_groups, keys=range(len(val_groups)))
    
    feature_cols = [col for col in train_concat.columns if col not in columns_to_drop]
    
    if scaling:
        scaler = StandardScaler()
        train_concat[feature_cols] = scaler.fit_transform(train_concat[feature_cols])
        val_concat[feature_cols] = scaler.transform(val_concat[feature_cols])
    else:
        scaler = None
    
    train_groups_scaled = [train_concat.xs(i).reset_index(drop=True) for i in range(len(train_groups))]
    val_groups_scaled = [val_concat.xs(i).reset_index(drop=True) for i in range(len(val_groups))]
    
    # Verify distributions
    def get_band_dist(groups, band_dict):
        if not groups:
            return {'many': 0, 'medium': 0, 'few': 0}
        all_data = pd.concat(groups, ignore_index=True)
        bands = all_data['sideSlip'].apply(lambda x: band_dict.get(round(float(x), 1), 'unknown'))
        total = len(bands)
        return {
            'many': 100 * (bands == 'many').sum() / total,
            'medium': 100 * (bands == 'medium').sum() / total,
            'few': 100 * (bands == 'few').sum() / total
        }
    
    train_dist = get_band_dist(train_groups_scaled, band_dict)
    val_dist = get_band_dist(val_groups_scaled, band_dict)
    
    print(f"\n✅ RANDOM Split (should match test distribution better):")
    print(f"Train: Many={train_dist['many']:.1f}%, Medium={train_dist['medium']:.1f}%, Few={train_dist['few']:.1f}%")
    print(f"Val:   Many={val_dist['many']:.1f}%, Medium={val_dist['medium']:.1f}%, Few={val_dist['few']:.1f}%")
    
    # Create datasets
    train_dataset = create_concat_dataset_from_groups(train_groups_scaled, target, sequence_length, columns_to_drop)
    val_dataset = create_concat_dataset_from_groups(val_groups_scaled, target, sequence_length, columns_to_drop)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nTrain: {len(train_dataset):,} sequences | Val: {len(val_dataset):,} sequences")
    print("="*80 + "\n")
    
    return train_loader, val_loader, scaler, train_dataset, val_dataset


def load_test_data(
    test_path,
    scaler,
    columns_to_drop,
    sequence_length=20,
    threshold_value=3.0,
    threshold_column="vxCG",
    batchsize=256,
    seed=42
):
    """Load test data (same as V2)."""
    print("="*80)
    print("📂 Loading Test Data")
    print("="*80)
    
    test_files = sorted(glob.glob(f"{test_path}/test_part_*.csv"))
    print(f"\nFound {len(test_files)} test files\n")
    
    all_groups = []
    for file_path in test_files:
        df = pd.read_csv(file_path)
        
        if 'sideSlip' not in df.columns and 'sideSlip' in columns_to_drop:
            df['sideSlip'] = 0.0
        if 'vyCG' not in df.columns and 'vyCG' in columns_to_drop:
            df['vyCG'] = 0.0
        
        groups = apply_threshold_v1_logic(df, threshold_column, threshold_value, sequence_length)
        all_groups.extend(groups)
    
    # Concat → scale → separate
    test_concat = pd.concat(all_groups, keys=range(len(all_groups)))
    feature_cols = [col for col in test_concat.columns if col not in columns_to_drop]
    test_concat[feature_cols] = scaler.transform(test_concat[feature_cols])
    test_groups_scaled = [test_concat.xs(i).reset_index(drop=True) for i in range(len(all_groups))]
    
    test_dataset = create_unsupervised_concat_dataset_from_groups(
        test_groups_scaled, sequence_length, columns_to_drop
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✅ Test sequences: {len(test_dataset):,}")
    if len(test_dataset) == 49328:
        print("   ✅ Matches Kaggle requirement!")
    print("="*80 + "\n")
    
    return test_loader
