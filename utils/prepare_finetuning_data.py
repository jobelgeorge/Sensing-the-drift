import pandas as pd
import glob
import os
import numpy as np

# CONFIG
TRAIN_DIR = "datasets_challenge/train"
OUTPUT_FILE = "datasets_challenge/train_merged_finetune.csv"

# Columns to keep (Inputs + Target at the end)
# We add 'date' because standard time-series loaders often expect a time column
COLS = ['handwheelAngle', 'vxCG', 'yawRate', 'ayCG', 'sideSlip']

print(" Merging training files for Fine-Tuning...")
files = sorted(glob.glob(os.path.join(TRAIN_DIR, "train_part_*.csv")))

all_data = []
for f in files:
    df = pd.read_csv(f)
    
    # Filter Noise (Crucial: Don't train on stopped cars)
    # We only keep rows where car is moving > 3.0 m/s
    df = df[df['vxCG'] > 3.0].copy()
    
    if len(df) > 0:
        all_data.append(df[COLS])

# Concatenate
full_df = pd.concat(all_data, ignore_index=True)

# Add dummy 'date' column (Required by many time-series loaders)
# We generate hourly timestamps just to satisfy the format
start_date = pd.Timestamp("2024-01-01")
full_df.insert(0, 'date', [start_date + pd.Timedelta(hours=i) for i in range(len(full_df))])

# Save
full_df.to_csv(OUTPUT_FILE, index=False)
print(f" Saved merged dataset to {OUTPUT_FILE}")
print(f" Total Training Samples: {len(full_df)}")
print(f" Columns: {list(full_df.columns)}")
print(" NOTE: 'sideSlip' must be the LAST column.")