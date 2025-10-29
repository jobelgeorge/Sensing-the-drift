"""
Data loading utilities for vehicle sideslip prediction challenge.

Provides challenge-accurate data loading with proper temporal validation splitting
to prevent data leakage in time series regression.
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, ConcatDataset

from .pipeline_preprocessing import (
    ApplyThreshold,
    _ConcatDataFrames,
    _SeparateDataFrames,
    CreateConcatDataset,
    SkScalerWrapper
)
from .custom_datasets import sequential_dataset, unsupervised_sequential_dataset
from .reproducibility_utils import seed_worker


FEATURE_NAMES = [
    "handwheelAngle", "throttle", "brake", "vxCG", "axCG", "ayCG",
    "yawRate", "rollRate", "pitchRate", "rollAngle", "pitchAngle",
    "chassisAccelFL", "chassisAccelFR", "chassisAccelRL", "chassisAccelRR",
    "wheelAccelFL", "wheelAccelFR", "wheelAccelRL", "wheelAccelRR",
    "deflectionFL", "deflectionFR", "deflectionRR"
]

feature_names = FEATURE_NAMES


def load_data(
    train_path: Union[str, Path],
    train_ratio: float = 0.8,
    columns_to_standardize: Optional[List[str]] = None,
    columns_to_drop: Optional[List[str]] = None,
    sequence_length: int = 20,
    target: Optional[str] = None,
    threshold: bool = False,
    threshold_value: Optional[float] = None,
    threshold_column: Optional[str] = None,
    batchsize: int = 512,
    scaling: bool = True,
    seed: int = 42,
    shuffle: bool = True,
    min_csv_length: int = 100,
    verbose: bool = False
) -> Tuple[DataLoader, DataLoader, Pipeline]:
    """
    Load data with challenge-accurate validation split.
    
    For each CSV file, extracts ONE contiguous block as validation at a random
    position (seeded for reproducibility). Remaining data (before and after the
    validation block) becomes training data. This prevents temporal data leakage
    and matches the competition's test set structure.
    
    Args:
        train_path: Path to folder containing CSV files
        train_ratio: Fraction of each CSV for training (0.8 = 80% train, 20% val)
        columns_to_standardize: Features to standardize (default: FEATURE_NAMES)
        columns_to_drop: Columns to exclude from features (default: [target])
        sequence_length: Sliding window size for sequential samples
        target: Target column name (None for unsupervised learning)
        threshold: Whether to filter rows by threshold
        threshold_value: Minimum value for threshold column
        threshold_column: Column to apply threshold on
        batchsize: Batch size for DataLoader
        scaling: Whether to apply StandardScaler
        seed: Random seed for reproducible validation block positioning
        shuffle: Whether to shuffle training batches
        min_csv_length: Skip CSVs shorter than this after filtering
        verbose: Whether to print progress information
        
    Returns:
        train_dataloader: DataLoader for training set
        val_dataloader: DataLoader for validation set
        pipeline: Fitted sklearn Pipeline for test preprocessing
        
    Raises:
        ValueError: If no CSV files found or no data remains after filtering
        FileNotFoundError: If train_path does not exist
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training path does not exist: {train_path}")
    
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    
    if columns_to_standardize is None:
        columns_to_standardize = FEATURE_NAMES.copy()
    
    if columns_to_drop is None:
        columns_to_drop = [target] if target else []
    
    csv_files = sorted(glob.glob(os.path.join(train_path, '*.csv')))
    if not csv_files:
        raise ValueError(f"No CSV files found in {train_path}")
    
    train_dfs = []
    val_dfs = []
    csv_stats = []
    
    np.random.seed(seed)
    
    for csv_idx, csv_path in enumerate(csv_files, start=1):
        csv_name = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        
        if threshold and threshold_column and threshold_value is not None:
            if threshold_column not in df.columns:
                raise ValueError(
                    f"Threshold column '{threshold_column}' not found in {csv_name}"
                )
            df = df[df[threshold_column] > threshold_value].reset_index(drop=True)
        
        if len(df) < min_csv_length:
            continue
        
        total_len = len(df)
        val_size = int(total_len * (1 - train_ratio))
        
        csv_rng = np.random.RandomState(seed + csv_idx)
        max_start = total_len - val_size
        val_start = csv_rng.randint(0, max_start + 1)
        val_end = val_start + val_size
        
        train_before = df.iloc[:val_start].copy()
        val_block = df.iloc[val_start:val_end].copy()
        train_after = df.iloc[val_end:].copy()
        
        if len(train_before) >= sequence_length:
            train_dfs.append(train_before)
        
        if len(train_after) >= sequence_length:
            train_dfs.append(train_after)
        
        if len(val_block) >= sequence_length:
            val_dfs.append(val_block)
        
        if verbose:
            total_train = len(train_before) + len(train_after)
            print(f"{csv_name}: Train={total_train}, Val={len(val_block)}, "
                  f"Block=[{val_start}:{val_end}]")
        
        csv_stats.append({
            'csv': csv_name,
            'total_len': total_len,
            'val_start': val_start,
            'val_end': val_end,
            'val_start_pct': val_start / total_len * 100
        })
    
    if not train_dfs:
        raise ValueError(
            "No training data after splitting. Check threshold/min_csv_length settings."
        )
    if not val_dfs:
        raise ValueError(
            "No validation data after splitting. Check sequence_length setting."
        )
    
    if verbose:
        val_positions = [s['val_start_pct'] for s in csv_stats]
        print(f"\nVal block positions: min={min(val_positions):.1f}%, "
              f"mean={np.mean(val_positions):.1f}%, max={max(val_positions):.1f}%")
        print(f"Train segments: {len(train_dfs)}, Val segments: {len(val_dfs)}")
    
    ct = ColumnTransformer(
        [("stand", StandardScaler(), columns_to_standardize)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    
    pipeline_steps = [
        ('concat', _ConcatDataFrames()),
        ('stand', ct.set_output(transform="pandas")),
        ('separate', _SeparateDataFrames())
    ]
    
    if not scaling:
        pipeline_steps = [s for s in pipeline_steps if s[0] not in ['concat', 'stand', 'separate']]
    
    pipeline = Pipeline(pipeline_steps)
    
    if scaling:
        pipeline.fit(train_dfs)
        train_dfs_scaled = pipeline.transform(train_dfs)
        val_dfs_scaled = pipeline.transform(val_dfs)
    else:
        train_dfs_scaled = train_dfs
        val_dfs_scaled = val_dfs
    
    if target is None:
        train_datasets = [
            unsupervised_sequential_dataset(
                df=df,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop,
                dtype=torch.float32
            ) for df in train_dfs_scaled
        ]
        
        val_datasets = [
            unsupervised_sequential_dataset(
                df=df,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop,
                dtype=torch.float32
            ) for df in val_dfs_scaled
        ]
    else:
        train_datasets = [
            sequential_dataset(
                df=df,
                target=target,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop,
                dtype=torch.float32
            ) for df in train_dfs_scaled
        ]
        
        val_datasets = [
            sequential_dataset(
                df=df,
                target=target,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop,
                dtype=torch.float32
            ) for df in val_dfs_scaled
        ]
    
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    if verbose:
        total_samples = len(train_dataset) + len(val_dataset)
        print(f"\nTrain: {len(train_dataset):,}, Val: {len(val_dataset):,}, "
              f"Ratio: {len(val_dataset)/total_samples*100:.1f}%")
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_dataloader, val_dataloader, pipeline


def load_test_data(
    test_path: Union[str, Path],
    batchsize: int,
    pipeline: Pipeline,
    seed: int = 42,
    load_anonymized_data: bool = False,
    verbose: bool = False
) -> DataLoader:
    """
    Load test data using pre-fitted preprocessing pipeline.
    
    Args:
        test_path: Path to folder containing test CSV files
        batchsize: Batch size for DataLoader
        pipeline: Pre-fitted preprocessing pipeline from load_data()
        seed: Random seed for reproducibility
        load_anonymized_data: If True, adds dummy target columns for anonymized test data
        verbose: Whether to print progress information
        
    Returns:
        test_dataloader: DataLoader for test set
        
    Raises:
        FileNotFoundError: If test_path does not exist
        ValueError: If no CSV files found
    """
    test_path = Path(test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test path does not exist: {test_path}")
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    csv_files = sorted(glob.glob(os.path.join(test_path, '*.csv')))
    if not csv_files:
        raise ValueError(f"No CSV files found in {test_path}")
    
    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        
        if load_anonymized_data:
            df['sideSlip'] = 0.0
            df['vyCG'] = 0.0
        
        dfs.append(df)
    
    test_dataset = pipeline.transform(dfs)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0
    )
    
    if verbose:
        print(f"Test dataset: {len(test_dataset):,} samples")
    
    return test_dataloader