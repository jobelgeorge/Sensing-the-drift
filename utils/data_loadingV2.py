"""
Data Loading V5: Stratified Temporal Split (V1-Compatible)
============================================================
V1-compatible processing with stratified temporal splits by band.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import glob
from pathlib import Path
import sys

# Import band mapping utilities
try:
    from utils.band_mapping import load_band_mapping
    from utils.custom_datasets import sequential_dataset, unsupervised_sequential_dataset
except ModuleNotFoundError:
    from band_mapping import load_band_mapping
    from custom_datasets import sequential_dataset, unsupervised_sequential_dataset


def apply_threshold_v1_logic(df, threshold_column="vxCG", threshold_value=3.0, seq_length=20):
    """
    Apply threshold using V1 pipeline logic: group consecutive sequences.
    This replicates ApplyThreshold transformer behavior.
    
    Returns list of DataFrames (valid groups longer than seq_length).
    """
    df_mask = df[threshold_column] > threshold_value
    groups = df[df_mask].groupby((~df_mask).cumsum())
    
    valid_groups = []
    for _, group in groups:
        if len(group) > seq_length:
            valid_groups.append(group.reset_index(drop=True))
    
    return valid_groups


def stratified_temporal_split_groups(groups_list, band_dict, val_ratio=0.15):
    """Split groups by band, last val_ratio% of each band's groups to validation."""
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
    
    # Temporal split per band
    for band_name in ['many', 'medium', 'few']:
        band_group_list = band_groups[band_name]
        
        if len(band_group_list) == 0:
            continue
        
        split_idx = int(len(band_group_list) * (1 - val_ratio))
        train_groups.extend([g[1] for g in band_group_list[:split_idx]])
        val_groups.extend([g[1] for g in band_group_list[split_idx:]])
    
    return train_groups, val_groups


def create_concat_dataset_from_groups(groups_list, target, seq_length, columns_to_drop):
    """Create ConcatDataset from groups - each group becomes separate dataset."""
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
    # Add X_columns attribute for notebook compatibility
    if len(datasets) > 0:
        concat_ds.X_columns = datasets[0].X_columns
    return concat_ds


def create_unsupervised_concat_dataset_from_groups(groups_list, seq_length, columns_to_drop):
    """Create ConcatDataset from groups for test data (no targets)."""
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
    val_ratio=0.15,
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
    Load and prepare data with stratified temporal split.
    
    V5 FIXED Features:
    - Maintains V1 pipeline group-level processing
    - Groups keep their boundaries (no cross-group sequences)
    - Temporal ordering within each band preserved
    - Identical train/val band distributions
    - Validation uses last 15% of each band's GROUPS
    
    Args:
        train_path: Path to training data directory
        val_ratio: Validation split ratio (default 0.15 = 15%)
        columns_to_drop: Columns to exclude from features
        sequence_length: Length of input sequences
        target: Target column name
        threshold: Whether to apply threshold filtering
        threshold_value: Minimum value for threshold
        threshold_column: Column to threshold on
        batchsize: Batch size for DataLoader
        scaling: Whether to apply StandardScaler
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle training batches
    
    Returns:
        train_loader, val_loader, scaler, train_dataset, val_dataset
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    band_dict = load_band_mapping()
    
    print("="*80)
    print("V5 Data Loader: Stratified Temporal Split")
    print("="*80)
    
    train_files = sorted(glob.glob(f"{train_path}/train_part_*.csv"))
    print(f"\nProcessing {len(train_files)} files...")
    
    all_groups = []
    for file_path in train_files:
        df = pd.read_csv(file_path)
        groups = apply_threshold_v1_logic(df, threshold_column, threshold_value, sequence_length)
        all_groups.extend(groups)
    
    print(f"Total groups after threshold: {len(all_groups)}")
    
    # Split groups by band
    train_groups, val_groups = stratified_temporal_split_groups(all_groups, band_dict, val_ratio)
    
    # V1 pipeline: concat → scale → separate
    train_concat = pd.concat(train_groups, keys=range(len(train_groups)))
    val_concat = pd.concat(val_groups, keys=range(len(val_groups)))
    
    feature_cols = [col for col in train_concat.columns if col not in columns_to_drop]
    
    if scaling:
        scaler = StandardScaler()
        train_concat[feature_cols] = scaler.fit_transform(train_concat[feature_cols])
        val_concat[feature_cols] = scaler.transform(val_concat[feature_cols])
    else:
        scaler = None
    
    # Separate back into groups
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
    
    print(f"\nTrain: Many={train_dist['many']:.1f}%, Medium={train_dist['medium']:.1f}%, Few={train_dist['few']:.1f}%")
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
    """
    Load test data with V1-compatible pipeline.
    
    Process:
    1. Apply threshold per file → get groups
    2. Concat groups with multi-index keys
    3. Scale using training scaler
    4. Separate back into groups
    5. Create ConcatDataset from groups
    
    This matches V1's exact pipeline flow.
    """
    print("="*80)
    print("📂 Loading Test Data (V1-Compatible)")
    print("="*80)
    
    test_files = sorted(glob.glob(f"{test_path}/test_part_*.csv"))
    print(f"\nFound {len(test_files)} test files")
    print(f"🔧 Applying threshold per file: {threshold_column} > {threshold_value} (grouping)\n")
    
    all_groups = []
    raw_count = 0
    
    for file_path in test_files:
        file_name = Path(file_path).name
        df = pd.read_csv(file_path)
        raw_count += len(df)
        
        # Add dummy columns if needed (will be dropped)
        if 'sideSlip' not in df.columns and 'sideSlip' in columns_to_drop:
            df['sideSlip'] = 0.0
        if 'vyCG' not in df.columns and 'vyCG' in columns_to_drop:
            df['vyCG'] = 0.0
        
        # Apply threshold grouping (V1 ApplyThreshold logic)
        groups = apply_threshold_v1_logic(df, threshold_column, threshold_value, sequence_length)
        
        if len(groups) > 0:
            total_samples = sum(len(g) for g in groups)
            print(f"   {file_name:25s} | {len(groups)} groups, {total_samples:,} samples")
            all_groups.extend(groups)
        else:
            print(f"   {file_name:25s} | No valid groups")
    
    print(f"\n✅ Total groups: {len(all_groups)}")
    
    # V1 pipeline: concat → scale → separate
    test_concat = pd.concat(all_groups, keys=range(len(all_groups)))
    after_threshold = len(test_concat)
    
    feature_cols = [col for col in test_concat.columns if col not in columns_to_drop]
    test_concat[feature_cols] = scaler.transform(test_concat[feature_cols])
    test_groups_scaled = [test_concat.xs(i).reset_index(drop=True) for i in range(len(all_groups))]
    
    print(f"\n📊 Test Data Summary:")
    print(f"   Raw samples: {raw_count:,}")
    print(f"   After threshold: {after_threshold:,} (removed {raw_count - after_threshold:,})")
    print(f"   Expected sequences: {after_threshold - sequence_length:,} (Kaggle expects 49,328)")
    
    # Create dataset from groups
    test_dataset = create_unsupervised_concat_dataset_from_groups(
        test_groups_scaled, sequence_length, columns_to_drop
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n✅ Test sequences: {len(test_dataset):,}")
    if len(test_dataset) == 49328:
        print("   ✅ Count matches Kaggle requirement!")
    else:
        print(f"   ⚠️  Expected 49,328 but got {len(test_dataset):,}")
    print("="*80 + "\n")
    
    return test_loader


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING V5 FIXED DATALOADER")
    print("="*80)
    
    train_loader, val_loader, scaler, train_ds, val_ds = load_data(
        train_path=r"datasets_challenge\train",
        val_ratio=0.15,
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
    )
    
    print("\n✅ Test completed successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    
    # Sample batch
    if len(train_loader) > 0:
        X_batch, y_batch = next(iter(train_loader))
        print(f"\n📦 Sample batch shape:")
        print(f"   X: {X_batch.shape}")
        print(f"   y: {y_batch.shape}")