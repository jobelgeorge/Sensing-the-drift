

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
    val_file: str,
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
    train_ratio: Optional[float] = None,  # Deprecated, kept for compatibility
) -> Tuple[DataLoader, DataLoader, Pipeline, Dict[str, float], Dict[str, float]]:
    """
    Load and preprocess data using dedicated validation file strategy.
    
    This function uses a complete file separation approach: one file is designated
    as the validation set while all other files are used for training.
    
    Args:
        train_path: Path to directory containing CSV files
        val_file: Filename (e.g., 'train_part_006.csv') to use as dedicated validation set.
                  This file will be EXCLUDED from training and used ONLY for validation.
        columns_to_standardize: Features to standardize (default: feature_names)
        columns_to_drop: Features to drop before creating sequences
        sequence_length: Length of sequences for datasets
        target: Target column name (None for unsupervised)
        threshold: Whether to apply threshold filtering
        threshold_value: Value for threshold
        threshold_column: Column to apply threshold on
        batchsize: Batch size for dataloaders
        scaling: Whether to apply standardization
        seed: Random seed
        shuffle: Whether to shuffle training data
        train_ratio: Deprecated parameter, kept for backwards compatibility (ignored)
    
    Returns:
        train_dataloader: Training dataloader (all files except val_file)
        val_dataloader: Validation dataloader (val_file only)
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
    
    # Dedicated validation file strategy
    print(f"\n{'='*70}")
    print(f"🎯 V2 Data Loader: Dedicated Validation File Strategy")
    print(f"{'='*70}")
    print(f"Validation file: {val_file}")
    print(f"{'='*70}\n")
    
    train_dfs = []
    val_dfs = []
    
    for filename in csv_files:
        file_name = os.path.basename(filename)
        df = pd.read_csv(filename)
        
        if file_name == val_file:
            print(f"✅ {file_name:25s} n={len(df):6,} → VALIDATION")
            val_dfs.append(df)
        else:
            print(f"🔵 {file_name:25s} n={len(df):6,} → TRAINING")
            train_dfs.append(df)
    
    if not val_dfs:
        raise ValueError(f"Validation file '{val_file}' not found in {train_path}")
    
    print(f"\n{'='*70}")
    print(f"📊 Split Summary:")
    print(f"   Training files:   {len(train_dfs):2d} files")
    print(f"   Validation files: {len(val_dfs):2d} file")
    print(f"{'='*70}\n")
    
    # === BUILD PIPELINE EXACTLY LIKE V1 ===
    ct = ColumnTransformer(
        [("stand", StandardScaler(), columns_to_standardize)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    
    # Build pipeline with same logic as V1
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
    
    # Add dataset creation step based on supervised vs unsupervised
    if target is None:
        # SSL (unsupervised)
        pipeline.steps.append([
            'concat_dataset',
            CreateConcatDataset(
                unsupervised_sequential_dataset,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop
            )
        ])
    else:
        # Supervised
        pipeline.steps.append([
            'concat_dataset',
            CreateConcatDataset(
                sequential_dataset,
                target=target,
                seq_length=sequence_length,
                columns_to_drop=columns_to_drop
            )
        ])
    
    # Remove steps based on flags (exactly like V1)
    if threshold == False:
        pipeline.steps = [step for step in pipeline.steps if step[0] not in ['threshold']]
    
    if scaling == False:
        pipeline.steps = [step for step in pipeline.steps if step[0] not in ['concat', 'stand', 'separate']]
    
    # === FIT AND TRANSFORM ON TRAIN DATA ===
    # This is where the magic happens:
    # - ApplyThreshold: List[DF] -> List[DF]
    # - _ConcatDataFrames: List[DF] -> Single DF with MultiIndex
    # - StandardScaler.fit(): Fits on the ENTIRE concatenated training data
    # - _SeparateDataFrames: Single DF -> List[DF]
    # - CreateConcatDataset: List[DF] -> PyTorch ConcatDataset
    train_dataset = pipeline.fit_transform(train_dfs)
    
    # === EXTRACT STATISTICS LIKE V1 ===
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
    
    # === TRANSFORM VALIDATION DATA (NO FITTING) ===
    val_dataset = pipeline.transform(val_dfs)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,  # Never shuffle validation
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=g
    )
    
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
    The pipeline was already fit on training data and must be reused as-is.
    
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
    
    # === ONLY TRANSFORM (NEVER FIT!) ===
    # The pipeline uses the SAME statistics fitted on training data:
    # - ApplyThreshold: uses same threshold
    # - StandardScaler: uses mean/std from training
    # - Everything else: same transformations
    test_dataset = pipeline.transform(dfs)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,  # Never shuffle test data
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return test_dataloader