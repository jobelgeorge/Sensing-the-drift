import pandas as pd
import numpy as np
import torch
import glob
import os
import sys
from sklearn.preprocessing import StandardScaler

# Ensure we can import your existing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from band_mapping import load_band_mapping, classify_band
    # Use your V4 logic to strip low-speed noise
    from data_loadingV4_file import apply_threshold_v1_logic
except ImportError:
    # Fallback if running from root
    from utils.band_mapping import load_band_mapping, classify_band
    from utils.data_loadingV4_file import apply_threshold_v1_logic

class TimePFNAdapter:
    def __init__(self, train_dir, history_len=96, feature_cols=None, target_col='sideSlip'):
        self.history_len = history_len
        self.feature_cols = feature_cols or ['handwheelAngle', 'vxCG', 'yawRate', 'ayCG']
        self.target_col = target_col
        self.band_dict = load_band_mapping()
        
        # --- 1. LOAD & FILTER TRAINING DATA ---
        print(" Loading and Filtering Training Data (vxCG > 3.0)...")
        files = sorted(glob.glob(os.path.join(train_dir, "train_part_*.csv")))
        valid_chunks = []
        
        for f in files:
            df = pd.read_csv(f)
            # Use V4 logic to get valid high-speed segments
            chunks = apply_threshold_v1_logic(
                df, 
                threshold_column="vxCG", 
                threshold_value=3.0, 
                seq_length=history_len + 1
            )
            valid_chunks.extend(chunks)
            
        self.full_df = pd.concat(valid_chunks, ignore_index=True)
        print(f" Clean Physics Pool: {len(self.full_df)} rows")

        # --- 2. FIT SCALER ---
        print(" Fitting StandardScaler...")
        self.scaler = StandardScaler()
        cols_to_scale = self.feature_cols + [self.target_col]
        self.full_df[cols_to_scale] = self.scaler.fit_transform(self.full_df[cols_to_scale])
        
        # --- 3. CLASSIFY BANDS ---
        # Inverse transform to get real degrees for classification
        raw_target = self.scaler.inverse_transform(self.full_df[cols_to_scale])[:, -1]
        self.bands = classify_band(raw_target, self.band_dict)
        
        # Create Indices
        self.indices_few = np.where((self.bands == 'few') | (self.bands == 'medium'))[0]
        self.indices_many = np.where(self.bands == 'many')[0]
        
        # Filter indices to ensure enough history exists
        self.indices_few = [i for i in self.indices_few if i >= history_len]
        self.indices_many = [i for i in self.indices_many if i >= history_len]
        
        print(f" Balancing Pool Ready: {len(self.indices_few)} Drift vs {len(self.indices_many)} Normal")

    def get_scaler(self):
        return self.scaler

    def get_balanced_context(self, n_drift=12, n_normal=52):
        """
        Creates the 'Prompt' for the model.
        Args:
            n_drift: Number of drift examples (Few/Medium shots)
            n_normal: Number of straight driving examples (Many shots)
        """
        # Sample from respective pools
        idx_drift = np.random.choice(self.indices_few, n_drift, replace=True)
        idx_normal = np.random.choice(self.indices_many, n_normal, replace=True)
        
        # Combine
        selected_indices = np.concatenate([idx_drift, idx_normal])
        # Shuffle them so the model doesn't see all drifts first
        np.random.shuffle(selected_indices)
        
        batch_x = [] 
        batch_y = [] 
        
        for idx in selected_indices:
            # Context X: History Features
            window = self.full_df.iloc[idx-self.history_len : idx][self.feature_cols].values
            # Context Y: Target History
            target_seq = self.full_df.iloc[idx-self.history_len : idx][self.target_col].values
            
            batch_x.append(window)
            batch_y.append(target_seq)
            
        # TimePFN Shapes: X=(Batch, Time, Feat), Y=(Batch, Time, 1)
        ctx_x = torch.tensor(np.array(batch_x), dtype=torch.float32)
        ctx_y = torch.tensor(np.array(batch_y), dtype=torch.float32).unsqueeze(-1) 
        
        return ctx_x, ctx_y