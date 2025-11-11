"""
DataLoader V4: Temporal Validation Split Strategy

CRITICAL FIX for V3's over-promising issue:
- V3 randomly shuffles each file before splitting → validation too similar to training
- V4 uses TEMPORAL split: last N files for validation (no shuffle)
- This better simulates test set (new driving scenarios/maneuvers)
- More realistic validation performance estimate

Key Differences:
- V1: Data leakage (scaler sees all data)
- V2: Single validation file (only 2% of data)
- V3: Random shuffle per file (over-optimistic validation)
- V4: TEMPORAL split by files (realistic validation) ✅
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import torch
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from .pipeline_preprocessing import (
    ApplyThreshold,
    _ConcatDataFrames,
    _SeparateDataFrames,
    CreateConcatDataset,
)
from .custom_datasets import sequential_dataset, unsupervised_sequential_dataset
from .reproducibility_utils import seed_worker


feature_names = ["handwheelAngle",
                 "throttle",
                 "brake",
                 "vxCG",
                 "axCG",
                 "ayCG",
                 "yawRate",
                 "rollRate",
                 "pitchRate",
                 "rollAngle",
                 "pitchAngle",
                 "chassisAccelFL",
                 "chassisAccelFR",
                 "chassisAccelRL",
                 "chassisAccelRR",
                 "wheelAccelFL",
                 "wheelAccelFR",
                 "wheelAccelRL",
                 "wheelAccelRR",
                 "deflectionFL",
                 "deflectionFR",
                 "deflectionRR"]


def load_data(
    train_path: Union[str, Path],
    val_files: Optional[Union[int, List[int]]] = None,
    columns_to_standardize: Optional[List[str]] = None,
    columns_to_drop: Optional[List[str]] = None,
    sequence_length: int = 20,
    target: Optional[str] = None,
    threshold: bool = True,
    threshold_value: float = 3.0,
    threshold_column: str = "vxCG",
    batchsize: int = 512,
    scaling: bool = True,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, Pipeline, Dict[str, float], Dict[str, float]]:
    """
    Load data with OPTIMIZED validation split using specific files.
    
    Key Innovation V4.1: 
    - Accepts SPECIFIC file indices (e.g., [9, 1, 6]) for validation
    - Uses scientifically-chosen files with best quality + representativeness
    - Falls back to temporal split if integer provided
    
    This ensures:
    1. Validation uses highest-quality, most representative files
    2. More realistic performance estimate (fixes V3's over-promising)
    3. Better generalization assessment
    4. No data leakage (scaler fit only on training files)
    
    Args:
        train_path: Path to directory containing CSV files
        val_files: EITHER:
                   - List[int]: Specific file indices for validation (e.g., [9, 1, 6])
                   - int: Number of LAST files for validation (temporal split)
                   - None: Defaults to [9, 1, 6] (optimal files from analysis)
        columns_to_standardize: Features to standardize (default: feature_names)
        columns_to_drop: Features to drop before creating sequences
        sequence_length: Length of sequences for datasets
        target: Target column name (None for unsupervised)
        threshold: Whether to apply threshold filtering (CRITICAL for removing low-speed noise!)
        threshold_value: Speed threshold in m/s (default: 3.0, removes stationary/crawling data)
        threshold_column: Column to threshold (default: vxCG = longitudinal velocity)
        batchsize: Batch size for dataloaders
        scaling: Whether to apply standardization
        seed: Random seed
        shuffle: Whether to shuffle training data (but NOT validation!)
    
    Returns:
        train_dataloader: Training dataloader (first files)
        val_dataloader: Validation dataloader (last files)
        pipeline: Fitted preprocessing pipeline (fitted on training data only)
        mean_dict: Dictionary of feature means from training data
        std_dict: Dictionary of feature stds from training data
    """
    # Set defaults
    if columns_to_standardize is None:
        columns_to_standardize = feature_names.copy()
    
    if columns_to_drop is None:
        columns_to_drop = []
    
    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Load all CSVs
    csv_files = sorted(glob.glob(os.path.join(train_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in {train_path}")
    
    # Default to optimal files [9, 1, 6] if not specified
    if val_files is None:
        val_files = [9, 1, 6]
        print("📌 Using DEFAULT optimal validation files: [9, 1, 6]")
    
    # Handle list of specific file indices vs count
    if isinstance(val_files, list):
        # Specific file indices provided
        val_file_indices = val_files
        if max(val_file_indices) >= len(csv_files):
            raise ValueError(f"File index {max(val_file_indices)} out of range (max: {len(csv_files)-1})")
        if min(val_file_indices) < 0:
            raise ValueError(f"File index {min(val_file_indices)} is negative")
        
        print(f"\n{'='*80}")
        print(f"🎯 V4.1 Data Loader: OPTIMIZED Validation Split (Specific Files)")
        print(f"{'='*80}")
        print(f"Total files: {len(csv_files)}")
        print(f"Validation file indices: {val_file_indices}")
        print(f"Training files: {len(csv_files) - len(val_file_indices)} (all others)")
        print(f"Validation files: {len(val_file_indices)} (scientifically chosen)")
        print(f"Threshold: {threshold_column} > {threshold_value} m/s" if threshold else "No threshold")
        print(f"{'='*80}\n")
        
        # Split by specific indices
        val_files_list = [csv_files[i] for i in val_file_indices]
        train_files = [f for i, f in enumerate(csv_files) if i not in val_file_indices]
        
    else:
        # Count from end (temporal split)
        if val_files >= len(csv_files):
            raise ValueError(f"val_files ({val_files}) must be less than total files ({len(csv_files)})")
        
        print(f"\n{'='*80}")
        print(f"🎯 V4 Data Loader: TEMPORAL Validation Split Strategy")
        print(f"{'='*80}")
        print(f"Total files: {len(csv_files)}")
        print(f"Training files: {len(csv_files) - val_files} (first files)")
        print(f"Validation files: {val_files} (LAST files - new temporal patterns)")
        print(f"Threshold: {threshold_column} > {threshold_value} m/s" if threshold else "No threshold")
        print(f"{'='*80}\n")
        
        # Split files temporally
        train_files = csv_files[:-val_files]
        val_files_list = csv_files[-val_files:]
    
    train_dfs = []
    val_dfs = []
    
    total_train_samples = 0
    total_val_samples = 0
    
    # Load training files
    print("📁 TRAINING FILES:")
    for filename in train_files:
        file_name = os.path.basename(filename)
        df = pd.read_csv(filename)
        
        # Apply threshold BEFORE splitting (removes low-speed noise)
        if threshold:
            original_len = len(df)
            df = df[df[threshold_column] > threshold_value].reset_index(drop=True)
            filtered_len = len(df)
            kept_pct = 100 * filtered_len / original_len if original_len > 0 else 0
        else:
            filtered_len = len(df)
            kept_pct = 100.0
        
        train_dfs.append(df)
        total_train_samples += len(df)
        
        print(f"   {file_name:25s} | "
              f"Samples: {filtered_len:6,} ({kept_pct:5.1f}% kept)")
    
    # Load validation files
    print(f"\n📁 VALIDATION FILES:")
    for idx, filename in enumerate(val_files_list):
        file_name = os.path.basename(filename)
        file_idx = csv_files.index(filename)
        df = pd.read_csv(filename)
        
        # Apply threshold
        if threshold:
            original_len = len(df)
            df = df[df[threshold_column] > threshold_value].reset_index(drop=True)
            filtered_len = len(df)
            kept_pct = 100 * filtered_len / original_len if original_len > 0 else 0
        else:
            filtered_len = len(df)
            kept_pct = 100.0
        
        val_dfs.append(df)
        total_val_samples += len(df)
        
        print(f"   [{file_idx:2d}] {file_name:25s} | "
              f"Samples: {filtered_len:6,} ({kept_pct:5.1f}% kept)")
    
    print(f"\n{'='*80}")
    print(f"📊 Split Summary:")
    print(f"   Training:   {len(train_dfs):2d} files, {total_train_samples:7,} samples")
    print(f"   Validation: {len(val_dfs):2d} files, {total_val_samples:7,} samples")
    print(f"   Val ratio:  {100*total_val_samples/(total_train_samples+total_val_samples):.1f}%")
    print(f"{'='*80}\n")
    
    # Build pipeline
    ct = ColumnTransformer(
        [("stand", StandardScaler(), columns_to_standardize)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    
    pipeline = Pipeline([
        ('threshold', ApplyThreshold(
            threshold=threshold_value,
            by=threshold_column,
            seq_length=sequence_length
        )),
        ('concat', _ConcatDataFrames()),
        ('stand', ct.set_output(transform="pandas")),
        ('separate', _SeparateDataFrames())
    ])
    
    # Add dataset creation step
    if target is None:
        pipeline.steps.append([
            'concat_dataset',
            CreateConcatDataset(
                unsupervised_sequential_dataset,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop
            )
        ])
    else:
        pipeline.steps.append([
            'concat_dataset',
            CreateConcatDataset(
                sequential_dataset,
                target=target,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop
            )
        ])
    
    # Remove steps based on flags
    if threshold == False:
        pipeline.steps = [step for step in pipeline.steps if step[0] not in ['threshold']]
    
    if scaling == False:
        pipeline.steps = [step for step in pipeline.steps if step[0] not in ['concat', 'stand', 'separate']]
    
    # FIT pipeline on TRAINING data only
    print("🔧 Fitting pipeline on training data...")
    train_dataset = pipeline.fit_transform(train_dfs)
    
    # Extract statistics
    if scaling == False:
        mean_dict = {}
        std_dict = {}
    else:
        mean_dict = {}
        std_dict = {}
        mean = ct.transformers_[0][1].mean_.tolist()
        std = ct.transformers_[0][1].scale_.tolist()
        for idx, col in enumerate(columns_to_standardize):
            mean_dict[col] = mean[idx]
            std_dict[col] = std[idx]
    
    # TRANSFORM validation data (using training statistics)
    print("🔧 Transforming validation data...")
    val_dataset = pipeline.transform(val_dfs)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,  # NEVER shuffle validation!
        worker_init_fn=seed_worker,
        generator=g
    )
    
    print(f"✅ Pipeline fitted and datasets created!")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples:   {len(val_dataset):,}\n")
    
    return train_dataloader, val_dataloader, pipeline, mean_dict, std_dict


def load_test_data(
    test_path: Union[str, Path],
    batchsize: int,
    pipeline: Pipeline,
    seed: int = 42,
    load_anonymized_data: bool = False,
) -> DataLoader:
    """
    Load and preprocess test data using the FITTED pipeline from training.
    
    CRITICAL: This function ONLY calls pipeline.transform() (never fit()).
    
    Args:
        test_path: Path to directory containing test CSV files
        batchsize: Batch size for dataloader
        pipeline: Fitted preprocessing pipeline from load_data()
        seed: Random seed
        load_anonymized_data: Whether to add dummy columns for anonymized data
    
    Returns:
        test_dataloader: Test dataloader
    """
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Load test CSVs
    dfs = []
    for filename in sorted(glob.glob(os.path.join(test_path, "*.csv"))):
        print(filename)
        df = pd.read_csv(filename)
        
        # Add missing columns with dummy values (they will be dropped by the pipeline)
        if load_anonymized_data:
            df['sideSlip'] = 0.0
            df['vyCG'] = 0.0
        
        dfs.append(df)
    
    # ONLY TRANSFORM (never fit!)
    test_dataset = pipeline.transform(dfs)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return test_dataloader
