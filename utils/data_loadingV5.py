"""
Data Loading V5: Stratified Temporal Split
==========================================
Splits data with temporal ordering while maintaining identical band distributions.

Key Features:
- Temporal ordering preserved within each band
- Identical train/val band distributions (no mismatch!)
- All files contribute to both train and validation
- Last 15% of each band goes to validation
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import glob
from pathlib import Path
import sys

# Import band mapping utilities
try:
    from utils.band_mapping import load_band_mapping
except ModuleNotFoundError:
    # Running as script from utils directory
    from band_mapping import load_band_mapping


class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series data with sequences.
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, target, sequence_length=20):
    """
    Create sequences from time series data.
    
    Args:
        data: Input features (N, F)
        target: Target values (N,)
        sequence_length: Length of each sequence
    
    Returns:
        X: Sequences (N-seq_len+1, seq_len, F)
        y: Targets (N-seq_len+1,)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i + sequence_length])
        y.append(target[i + sequence_length - 1])
    
    return np.array(X), np.array(y)


def stratified_temporal_split(df, band_dict, val_ratio=0.15, threshold_column="vxCG", threshold_value=3.0):
    """
    Split data with temporal ordering while maintaining identical band distributions.
    
    Strategy:
    1. Apply threshold filter
    2. For each band (many/medium/few):
       - Find all samples belonging to this band (in temporal order)
       - Take first (1-val_ratio) for training
       - Take last val_ratio for validation
    3. Concatenate all train/val indices
    
    This ensures:
    - Train and Val have IDENTICAL band distributions
    - Temporal ordering preserved (validation is "future" data)
    - All bands represented proportionally in both sets
    
    Args:
        df: DataFrame with sideslip targets
        band_dict: Dictionary mapping sideslip values to bands
        val_ratio: Fraction for validation (default 0.15 = 15%)
        threshold_column: Column to threshold
        threshold_value: Minimum value to keep
    
    Returns:
        train_indices, val_indices: Lists of indices for train/val splits
    """
    # Apply threshold
    df_filtered = df[df[threshold_column] > threshold_value].copy()
    
    if len(df_filtered) == 0:
        return [], []
    
    # Classify each sample by band (round to 1 decimal for lookup)
    df_filtered['band'] = df_filtered['sideSlip'].apply(
        lambda x: band_dict.get(round(float(x), 1), 'unknown')
    )
    
    train_indices = []
    val_indices = []
    
    # Split each band temporally
    for band_name in ['many', 'medium', 'few']:
        # Get indices for this band (already in temporal order)
        band_mask = df_filtered['band'] == band_name
        band_indices = df_filtered[band_mask].index.tolist()
        
        if len(band_indices) == 0:
            continue
        
        # Calculate split point (first 85%, last 15%)
        split_idx = int(len(band_indices) * (1 - val_ratio))
        
        # Split temporally
        train_indices.extend(band_indices[:split_idx])   # First 85%
        val_indices.extend(band_indices[split_idx:])      # Last 15%
    
    return train_indices, val_indices


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
    
    V5 Features:
    - Maintains temporal ordering within each band
    - Identical train/val band distributions
    - All files contribute to both train and validation
    - Validation uses last 15% of each band from each file
    
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
    
    # Load band mapping
    band_dict = load_band_mapping()
    
    print("="*80)
    print("🎯 V5 Data Loader: Stratified Temporal Split")
    print("="*80)
    print(f"Validation ratio: {val_ratio*100:.0f}% per band per file")
    print(f"Threshold: {threshold_column} > {threshold_value}")
    print(f"Temporal ordering: PRESERVED within each band")
    print(f"Distribution matching: EXACT (train = val)")
    print("="*80)
    
    # Get all training files
    train_files = sorted(glob.glob(f"{train_path}/train_part_*.csv"))
    print(f"\nTotal files: {len(train_files)}")
    
    # Collect all data with stratified temporal splits
    all_train_data = []
    all_val_data = []
    
    total_train_samples = 0
    total_val_samples = 0
    
    print("\n📁 Processing files with stratified temporal split...")
    for file_path in train_files:
        file_name = Path(file_path).name
        df = pd.read_csv(file_path)
        
        # Get stratified temporal split for this file
        train_indices, val_indices = stratified_temporal_split(
            df, band_dict, val_ratio, threshold_column, threshold_value
        )
        
        if len(train_indices) == 0 and len(val_indices) == 0:
            print(f"   {file_name:25s} | Skipped (no samples after threshold)")
            continue
        
        # Extract train and val data
        if len(train_indices) > 0:
            all_train_data.append(df.loc[train_indices])
            total_train_samples += len(train_indices)
        
        if len(val_indices) > 0:
            all_val_data.append(df.loc[val_indices])
            total_val_samples += len(val_indices)
        
        train_pct = 100 * len(train_indices) / len(df) if len(df) > 0 else 0
        val_pct = 100 * len(val_indices) / len(df) if len(df) > 0 else 0
        
        print(f"   {file_name:25s} | Train: {len(train_indices):5,} ({train_pct:4.1f}%) | Val: {len(val_indices):5,} ({val_pct:4.1f}%)")
    
    # Concatenate all data
    train_df = pd.concat(all_train_data, ignore_index=True)
    val_df = pd.concat(all_val_data, ignore_index=True)
    
    print("\n" + "="*80)
    print("📊 Split Summary:")
    print(f"   Training:   {total_train_samples:6,} samples ({100*total_train_samples/(total_train_samples+total_val_samples):.1f}%)")
    print(f"   Validation: {total_val_samples:6,} samples ({100*total_val_samples/(total_train_samples+total_val_samples):.1f}%)")
    print("="*80)
    
    # Verify band distributions
    print("\n📊 Verifying Band Distributions...")
    
    def get_band_dist(df, band_dict):
        bands = df['sideSlip'].apply(lambda x: band_dict.get(round(float(x), 1), 'unknown'))
        total = len(bands)
        return {
            'many': 100 * (bands == 'many').sum() / total,
            'medium': 100 * (bands == 'medium').sum() / total,
            'few': 100 * (bands == 'few').sum() / total
        }
    
    train_dist = get_band_dist(train_df, band_dict)
    val_dist = get_band_dist(val_df, band_dict)
    
    print(f"\n🔵 TRAIN: Many={train_dist['many']:.1f}%, Medium={train_dist['medium']:.1f}%, Few={train_dist['few']:.1f}%")
    print(f"🟢 VAL:   Many={val_dist['many']:.1f}%, Medium={val_dist['medium']:.1f}%, Few={val_dist['few']:.1f}%")
    
    # Check mismatch
    max_diff = max(
        abs(train_dist['many'] - val_dist['many']),
        abs(train_dist['medium'] - val_dist['medium']),
        abs(train_dist['few'] - val_dist['few'])
    )
    
    if max_diff < 0.1:
        print(f"✅ PERFECT MATCH! Maximum difference: {max_diff:.3f}%")
    else:
        print(f"⚠️  Distribution difference: {max_diff:.1f}%")
    
    # Prepare features and targets
    print("\n🔧 Preparing features and targets...")
    
    X_train = train_df.drop(columns=columns_to_drop).values
    y_train = train_df[target].values
    
    X_val = val_df.drop(columns=columns_to_drop).values
    y_val = val_df[target].values
    
    # Create sequences
    print(f"🔧 Creating sequences (length={sequence_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    
    # Scaling
    if scaling:
        print("🔧 Fitting scaler on training data...")
        scaler = StandardScaler()
        
        # Flatten sequences for scaling
        n_train, seq_len, n_features = X_train_seq.shape
        X_train_flat = X_train_seq.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_train_seq = X_train_scaled.reshape(n_train, seq_len, n_features)
        
        # Transform validation
        n_val, _, _ = X_val_seq.shape
        X_val_flat = X_val_seq.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_flat)
        X_val_seq = X_val_scaled.reshape(n_val, seq_len, n_features)
    else:
        scaler = None
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=shuffle,  # Shuffle batches, but data already stratified
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,  # Never shuffle validation
        num_workers=0,
        pin_memory=True
    )
    
    print("✅ Data loading complete!")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples:   {len(val_dataset):,}")
    print(f"   Batch size:    {batchsize}")
    print(f"   Features:      {X_train_seq.shape[-1]}")
    print("="*80 + "\n")
    
    return train_loader, val_loader, scaler, train_dataset, val_dataset


def load_test_data(
    test_path,
    scaler,
    columns_to_drop,
    sequence_length=20,
    batchsize=256
):
    """
    Load and preprocess test data with same pipeline as training.
    
    IMPORTANT: Test data should NOT have threshold applied!
    Kaggle evaluates on all test samples after sequencing.
    
    Args:
        test_path: Path to test directory
        scaler: Fitted scaler from training data
        columns_to_drop: List of columns to drop (should match training)
        sequence_length: Length of sequences to create
        batchsize: Batch size for DataLoader
        
    Returns:
        test_loader: DataLoader for test data
        test_ids: Array of sequential test IDs (0 to N-1)
        test_dataset: PyTorch Dataset
    """
    print("="*80)
    print("📂 LOADING TEST DATA (NO THRESHOLD)")
    print("="*80)
    
    # Load all test files
    test_files = sorted(glob.glob(f"{test_path}/test_part_*.csv"))
    print(f"\nFound {len(test_files)} test files")
    
    all_test_data = []
    total_samples = 0
    
    for file_path in test_files:
        file_name = Path(file_path).name
        df = pd.read_csv(file_path)
        
        # Add dummy vyCG column if missing (will be dropped anyway)
        if 'vyCG' not in df.columns and 'vyCG' in columns_to_drop:
            df['vyCG'] = 0.0
        
        # Drop columns to match training
        df_features = df.drop(columns=columns_to_drop, errors='ignore')
        
        print(f"   {file_name}: {len(df):,} samples → {df_features.shape[1]} features")
        all_test_data.append(df_features.values)
        total_samples += len(df)
    
    # Concatenate all test data
    test_data = np.concatenate(all_test_data, axis=0)
    
    print(f"\n📊 Test Data Summary:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Features:      {test_data.shape[1]}")
    
    # Scale test data using training scaler
    print(f"\n🔧 Scaling test data using training scaler...")
    test_data_scaled = scaler.transform(test_data)
    
    # Create sequences
    print(f"🔧 Creating sequences (length={sequence_length})...")
    X_test_seq = []
    for i in range(len(test_data_scaled) - sequence_length + 1):
        X_test_seq.append(test_data_scaled[i:i + sequence_length])
    X_test_seq = np.array(X_test_seq)
    
    # Generate sequential IDs (0 to N-1)
    test_ids = np.arange(len(X_test_seq))
    
    print(f"✅ Created {len(X_test_seq):,} sequences")
    print(f"   Shape: {X_test_seq.shape}")
    print(f"   IDs: {test_ids[0]} to {test_ids[-1]}")
    
    # Create dataset (no targets for test data)
    test_dataset = TestDataset(X_test_seq)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,  # Never shuffle test data
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n✅ Test data loading complete!")
    print(f"   Test samples: {len(test_dataset):,}")
    print(f"   Batch size:   {batchsize}")
    print(f"   Features:     {X_test_seq.shape[-1]}")
    print("="*80 + "\n")
    
    return test_loader, test_ids, test_dataset


def create_sequences_with_ids(data, ids, sequence_length=20):
    """
    Create sequences from data while tracking IDs.
    
    Args:
        data: Input features (N, F)
        ids: Sample IDs (N,)
        sequence_length: Length of each sequence
        
    Returns:
        X: Sequences of shape (N-seq_len+1, seq_len, F)
        ids_seq: IDs corresponding to last element of each sequence
    """
    X = []
    ids_seq = []
    
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i+sequence_length])
        ids_seq.append(ids[i+sequence_length-1])  # ID of last element in sequence
    
    return np.array(X), np.array(ids_seq)


class TestDataset(Dataset):
    """
    Dataset for test data (no targets).
    """
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING V5 DATALOADER")
    print("="*80)
    
    # Test loading
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
    X_batch, y_batch = next(iter(train_loader))
    print(f"\n📦 Sample batch shape:")
    print(f"   X: {X_batch.shape}")
    print(f"   y: {y_batch.shape}")
