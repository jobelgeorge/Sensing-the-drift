import os
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, WeightedRandomSampler

# Assuming these are available in your local environment as per your snippet
from .pipeline_preprocessing import ApplyThreshold, _ConcatDataFrames, _SeparateDataFrames, CreateConcatDataset
from .custom_datasets import sequential_dataset, unsupervised_sequential_dataset
from .reproducibility_utils import seed_worker

feature_names = ["handwheelAngle", "throttle", "brake", "vxCG", "axCG", "ayCG", 
                 "yawRate", "rollRate", "pitchRate", "rollAngle", "pitchAngle", 
                 "chassisAccelFL", "chassisAccelFR", "chassisAccelRL", "chassisAccelRR", 
                 "wheelAccelFL", "wheelAccelFR", "wheelAccelRL", "wheelAccelRR", 
                 "deflectionFL", "deflectionFR", "deflectionRR"]

def load_data(
    train_path,
    dist_csv_path, # Pass the path to sideSlip_distribution.csv
    val_file_name = "train_part_017.csv", # Set as requested
    columns_to_standardize = feature_names,
    columns_to_drop = [],
    sequence_length = 20,
    target = 'sideSlip',
    threshold = False,
    threshold_value = None,
    threshold_column = None,
    batchsize = 512,
    scaling = True,
    seed = 42,
    shuffle = False,
    ):
    """
    Load and preprocess data with Leave-One-Out Validation and Weighted Sampling.
    """

    g = torch.Generator()
    g.manual_seed(seed)
    
    # 1. Load the distribution mapping for weight calculation
    dist_df = pd.read_csv(dist_csv_path)
    # Create mapping: sideSlip value (rounded) -> weight
    # Weights are inverse of frequency: Many (~1), Medium (~12), Few (~45)
    mapping_dict = dict(zip(dist_df['sideSlip'].round(1), dist_df['band']))
    
    alpha = 0.5  # try 0.25, 0.5, 0.75

    def get_weight_from_val(val):
        b = mapping_dict.get(round(float(val), 1), 'many')

        if b == 'few':
            base_weight = 50.0
        elif b == 'medium':
            base_weight = 15.0
        else:
            base_weight = 1.0

        # Apply exponent scaling
        return base_weight ** alpha
        

    # We remove the files that have near-zero variety
    files_to_exclude = []

    # 2. Separate files into Train and Validation
    train_dfs = []
    val_dfs = []
    
    all_files = sorted(glob.glob(os.path.join(train_path, '*.csv')))
    for filename in all_files:
        base_name = os.path.basename(filename)
        
        # Scenario A: This is our validation "Stress Test"
        if base_name == val_file_name:
            print(f"VALIDATION: Loading {base_name}")
            val_dfs.append(pd.read_csv(filename))
            
        # Scenario B: These are redundant/low-variety files we want to skip
        elif base_name in files_to_exclude:
            continue
            
        # Scenario C: Active driving files for training
        else:
            train_dfs.append(pd.read_csv(filename))
        # CRITICAL SAFETY CHECK:
    if len(train_dfs) == 0:
        raise ValueError(f"No training files found! Check if {train_path} is correct "
                        f"or if all files were excluded by val_file_name: {val_file_name}")

    # 3. Setup Preprocessing Pipeline
    ct = ColumnTransformer([("stand", StandardScaler(), columns_to_standardize)],
                           remainder="passthrough",
                           verbose_feature_names_out=False)

    pipeline = Pipeline([
        ('threshold', ApplyThreshold(threshold=threshold_value, by=threshold_column, seq_length=sequence_length)), 
        ('concat', _ConcatDataFrames()), 
        ('stand', ct.set_output(transform="pandas")),
        ('separate', _SeparateDataFrames())
    ])
    
    # 4. Define Dataset Creation Logic
    if target is None:
        # SSL (Unsupervised)
        dataset_class = unsupervised_sequential_dataset
        pipeline.steps.append(['concat dataset', CreateConcatDataset(dataset_class, seq_length=sequence_length, columns_to_drop=columns_to_drop)])
    else:
        # Supervised
        dataset_class = sequential_dataset
        pipeline.steps.append(['concat dataset', CreateConcatDataset(dataset_class, target=target, seq_length=sequence_length, columns_to_drop=columns_to_drop)])

    if not threshold:
        pipeline.steps = [step for step in pipeline.steps if step[0] != 'threshold']

    if not scaling:
        pipeline.steps = [step for step in pipeline.steps if step[0] not in ['concat', 'stand', 'separate']]
         
    # 5. Transform Data
    train_dataset = pipeline.fit_transform(train_dfs)
    
    # 6. Weighted Sampler for Train Dataset
    print("Calculating weights for WeightedRandomSampler...")
    
    train_targets = []
    valid_indices = []

    for i in range(len(train_dataset)):
        try:
            _, y = train_dataset[i]
            
            # Robust extraction of the target value
            if torch.is_tensor(y):
                if y.numel() == 0:
                    # Skip empty targets to avoid IndexError
                    continue
                elif y.dim() == 0:
                    # Handle scalar tensors: torch.tensor(0.5)
                    val = y.item()
                else:
                    # Handle sequences or 1D tensors: torch.tensor([0.5])
                    val = y[-1].item()
            else:
                # Handle raw floats/ints
                val = y
                
            train_targets.append(val)
            valid_indices.append(i)
            
        except Exception as e:
            # Silently skip items that cause errors during target extraction
            continue

    # If some samples were empty/invalid, we wrap the dataset in a Subset
    if len(valid_indices) < len(train_dataset):
        print(f"Note: Skipped {len(train_dataset) - len(valid_indices)} empty/invalid samples.")
        train_dataset = torch.utils.data.Subset(train_dataset, valid_indices)

    # Now calculate weights for the valid samples only
    sample_weights = [get_weight_from_val(t) for t in train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    print("Max weight used:", max(sample_weights))
    print(pd.Series(sample_weights).describe())


    # 7. Create Dataloaders
    # Note: When using a Sampler, shuffle must be False
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batchsize, 
        sampler=sampler, 
        worker_init_fn=seed_worker, 
        generator=g
    )

    if val_dfs:
        val_dataset = pipeline.transform(val_dfs)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batchsize, 
            shuffle=False, 
            worker_init_fn=seed_worker, 
            generator=g
        )
    else:
        val_dataloader = None
    
    return train_dataloader, val_dataloader, pipeline


def load_test_data(
    test_path,
    batchsize,
    pipeline,
    seed = 42,
    load_anonymized_data = False):

    g = torch.Generator()
    g.manual_seed(seed)
    
    dfs = []
    for filename in sorted(glob.glob(os.path.join(test_path, '*.csv'))):
        print(filename)
        df = pd.read_csv(filename)
        # Add missing columns with dummy values (they will be dropped by the pipeline)
        if load_anonymized_data:
            df['sideSlip'] = 0.0
            df['vyCG'] = 0.0
            df['sideSlip_rounded'] = 0.0
        dfs.append(df)

    test_dataset = pipeline.transform(dfs)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, worker_init_fn=seed_worker, generator=g)
    return test_dataloader