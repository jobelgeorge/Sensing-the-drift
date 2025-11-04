

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
    train_ratio: Optional[float] = None,
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
    val_file: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Pipeline, Dict[str, float], Dict[str, float]]:
    """
    Load and preprocess data with V1-style pipeline but V2-style temporal split.
    
    This matches V1's preprocessing exactly while fixing the data leakage issue
    by using temporal (contiguous) splits instead of random splits.
    
    Args:
        train_path: Path to directory containing CSV files
        train_ratio: Fraction for training (None = no validation split). Ignored if val_file is provided.
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
        val_file: Optional filename (e.g., 'train_part_006.csv') to use as dedicated validation set.
                  If provided, this file will be excluded from training and used for validation.
    
    Returns:
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader (None if train_ratio is None and val_file is None)
        pipeline: Fitted preprocessing pipeline
        mean_dict: Dictionary of feature means (empty if scaling=False)
        std_dict: Dictionary of feature stds (empty if scaling=False)
    """
    # Set defaults to match V1
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
    
    # If val_file is specified, use it as dedicated validation set
    if val_file is not None:
        print(f"\n=== Using dedicated validation file: {val_file} ===\n")
        train_dfs = []
        val_dfs = []
        
        for filename in csv_files:
            file_name = os.path.basename(filename)
            df = pd.read_csv(filename)
            print(f"{file_name}: n={len(df)}")
            
            if file_name == val_file:
                print(f"  -> Assigned to VALIDATION set")
                val_dfs.append(df)
            else:
                print(f"  -> Assigned to TRAINING set")
                train_dfs.append(df)
        
        if not val_dfs:
            raise ValueError(f"Validation file '{val_file}' not found in {train_path}")
        
        print(f"\nTraining files: {len(train_dfs)}, Validation files: {len(val_dfs)}")
    
    elif train_ratio is not None:
        # Original temporal split logic
        import numpy as np
        rng = np.random.RandomState(seed)
        
        dfs = []
        for filename in csv_files:
            print(filename)
            dfs.append(pd.read_csv(filename))
        
        train_dfs = []
        val_dfs = []
        
        for idx, (df, filename) in enumerate(zip(dfs, csv_files)):
            n = len(df)
            file_name = os.path.basename(filename)
            
            if n < 2:
                print(f" IGNORED: {file_name} (length={n}, too short)")
                continue
            
            val_size = int(n * (1 - train_ratio))
            buffer = sequence_length - 1
            max_start = n - val_size - buffer
            
            if max_start <= buffer:
                print(f" IGNORED: {file_name} (length={n}, insufficient for buffer={buffer})")
                continue
            
            val_start = rng.randint(buffer, max_start + 1)
            val_end = val_start + val_size
            val_position_pct = (val_start / n) * 100
            
            print(f"{file_name}: n={n}, val_block=[{val_start}:{val_end}] ")
            
            train_df = pd.concat([
                df.iloc[:val_start - buffer],
                df.iloc[val_end + buffer:]
            ]).reset_index(drop=True)
            
            val_df = df.iloc[val_start:val_end].reset_index(drop=True)
            
            train_dfs.append(train_df)
            val_dfs.append(val_df)
    else:
        # No validation split
        dfs = []
        for filename in csv_files:
            print(filename)
            dfs.append(pd.read_csv(filename))
        train_dfs = dfs
        val_dfs = None
    
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
    if val_dfs is not None:
        val_dataset = pipeline.transform(val_dfs)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batchsize,
            shuffle=False,  # Never shuffle validation
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        val_dataloader = None
    
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