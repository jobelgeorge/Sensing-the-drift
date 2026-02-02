"""
TimePFN Analysis for Sideslip Estimation
Baseline: 0.217 | TimePFN Best: 0.60
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Path setup
def find_project_root():
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "utils").exists() and (p / "models").exists():
            return p
    return here

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TimePFNConfig:
    """Model config matching fine-tuning script."""
    def __init__(self, seq_len=96, n_vars=5, device='cuda'):
        self.seq_len = seq_len
        self.pred_len = seq_len
        self.enc_in = n_vars
        self.dec_in = n_vars
        self.c_out = n_vars
        self.d_model = 1024
        self.d_ff = 512
        self.n_heads = 8
        self.e_layers = 8
        self.d_layers = 1
        self.dropout = 0.1
        self.patch_size = 16
        self.embed_dim = 256
        self.use_norm = True
        self.activation = 'gelu'
        self.output_attention = False
        self.embed = 'timeF'
        self.freq = 'h'
        self.factor = 1
        self.use_gpu = device == 'cuda'
        self.class_strategy = 'projection'
        self.num_class = 1
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.down_sampling_layers = 0
        self.down_sampling_method = 'avg'
        self.down_sampling_window = 1


class Config:
    TRAIN_PATH = PROJECT_ROOT / "datasets_challenge" / "train"
    TEST_PATH = PROJECT_ROOT / "datasets_challenge" / "test"
    TIMEPFN_ROOT = PROJECT_ROOT / "TimePFN"
    SEQ_LEN = 20
    CTX_LEN = 96
    THRESHOLD = 3.0
    EXPECTED_ROWS = 49328
    FEATURES = ['handwheelAngle', 'vxCG', 'yawRate', 'ayCG']
    CLAMP = 0.5
    SMOOTH_WIN = 11
    N_DRIFT = 12
    N_NORMAL = 52
    SEED = 42


class DataAdapter:
    """Prepare data for TimePFN."""
    def __init__(self, train_dir, ctx_len, features):
        from sklearn.preprocessing import StandardScaler
        self.ctx_len = ctx_len
        self.features = features
        
        train_files = sorted(glob.glob(str(Path(train_dir) / "train_part_*.csv")))
        self.df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
        
        self.scaler = StandardScaler()
        self.scaler.fit(self.df[features + ['sideSlip']].values)
    
    def get_context(self, n_drift, n_normal):
        df = self.df[self.df['vxCG'] > 3.0].copy()
        df['is_drift'] = np.abs(df['sideSlip']) > 2.0
        
        np.random.seed(Config.SEED)
        ctx_x, ctx_y = [], []
        
        for is_drift, n in [(True, n_drift), (False, n_normal)]:
            indices = df[df['is_drift'] == is_drift].index.tolist()
            for _ in range(min(n, len(indices) // self.ctx_len)):
                idx = np.random.choice(indices)
                start, end = max(0, idx - self.ctx_len + 1), idx + 1
                if end - start < self.ctx_len:
                    continue
                chunk = df.loc[start:end-1, self.features + ['sideSlip']].values
                if len(chunk) == self.ctx_len:
                    scaled = self.scaler.transform(chunk)
                    ctx_x.append(scaled[:, :-1])
                    ctx_y.append(scaled[:, -1:])
        
        return (torch.tensor(np.array(ctx_x), dtype=torch.float32),
                torch.tensor(np.array(ctx_y), dtype=torch.float32))


def apply_threshold_filter(df, threshold=3.0, seq_len=20):
    """Filter consecutive valid samples."""
    valid = df['vxCG'] > threshold
    groups, current = [], []
    for idx, is_valid in enumerate(valid):
        if is_valid:
            current.append(idx)
        else:
            if len(current) >= seq_len:
                groups.append(df.iloc[current].copy())
            current = []
    if len(current) >= seq_len:
        groups.append(df.iloc[current].copy())
    return groups


def postprocess(preds, clamp=0.5, window=11):
    """Zero-clamp and smooth predictions."""
    out = preds.copy()
    out[np.abs(out) < clamp] = 0.0
    try:
        out = savgol_filter(out, window_length=window, polyorder=2)
    except ValueError:
        pass
    out[np.abs(out) < clamp] = 0.0
    return out


def run_experiment():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load TimePFN
    if str(cfg.TIMEPFN_ROOT) not in sys.path:
        sys.path.insert(0, str(cfg.TIMEPFN_ROOT))
    
    try:
        from model.TimePFN import Model as TimePFN
    except ImportError:
        print(f"TimePFN not found at {cfg.TIMEPFN_ROOT}")
        return None
    
    n_vars = len(cfg.FEATURES) + 1
    model_cfg = TimePFNConfig(seq_len=cfg.CTX_LEN, n_vars=n_vars, device=device)
    model = TimePFN(model_cfg).to(device)
    
    # Load weights
    ft_ckpts = glob.glob(str(cfg.TIMEPFN_ROOT / "checkpoints/sensing_drift_ft_*/checkpoint.pth"))
    if ft_ckpts:
        model.load_state_dict(torch.load(max(ft_ckpts, key=os.path.getmtime), 
                                          map_location=device), strict=False)
    else:
        base = cfg.TIMEPFN_ROOT / "checkpoints/TimePFN/checkpoint.pth"
        if base.exists():
            model.load_state_dict(torch.load(str(base), map_location=device), strict=False)
    model.eval()
    
    # Prepare context
    adapter = DataAdapter(str(cfg.TRAIN_PATH), cfg.CTX_LEN, cfg.FEATURES)
    ctx_x, ctx_y = adapter.get_context(cfg.N_DRIFT, cfg.N_NORMAL)
    ctx_x, ctx_y = ctx_x.to(device), ctx_y.to(device)
    scaler = adapter.scaler
    
    # Inference
    test_files = sorted(glob.glob(str(cfg.TEST_PATH / "test_part_*.csv")))
    preds, ids = [], []
    global_id = 0
    
    for f in tqdm(test_files, desc="Processing"):
        df = pd.read_csv(f)
        groups = apply_threshold_filter(df, cfg.THRESHOLD, cfg.SEQ_LEN)
        
        for group in groups:
            group_scaled = group.copy()
            group_scaled['sideSlip'] = 0.0
            group_scaled[cfg.FEATURES + ['sideSlip']] = scaler.transform(
                group_scaled[cfg.FEATURES + ['sideSlip']])
            feats = group_scaled[cfg.FEATURES].values
            
            n_preds = len(group) - cfg.SEQ_LEN
            ids.extend(range(global_id, global_id + n_preds))
            global_id += n_preds
            
            for i in range(n_preds):
                t_idx = i + cfg.SEQ_LEN - 1
                w_end = t_idx + 1
                w_start = w_end - cfg.CTX_LEN
                
                if w_start < 0:
                    valid = feats[0:w_end]
                    pad = np.zeros((cfg.CTX_LEN - len(valid), len(cfg.FEATURES)))
                    window = np.vstack([pad, valid])
                else:
                    window = feats[w_start:w_end]
                
                x = torch.tensor(window).float().unsqueeze(0).to(device)
                y_dummy = torch.zeros((1, cfg.CTX_LEN, 1)).to(device)
                
                with torch.no_grad():
                    out = model(ctx_x, ctx_y, x, y_dummy)
                    preds.append(out[0, -1, -1].item())
    
    # Inverse transform
    dummy = np.zeros((len(preds), len(cfg.FEATURES)))
    matrix = np.hstack([dummy, np.array(preds).reshape(-1, 1)])
    raw_preds = scaler.inverse_transform(matrix)[:, -1]
    
    # Postprocess
    final_preds = postprocess(raw_preds, cfg.CLAMP, cfg.SMOOTH_WIN)
    
    # Save
    submission = pd.DataFrame({"id": ids, "predicted_sideslip": final_preds})
    submission.to_csv("submission_timepfn.csv", index=False)
    
    print(f"\nPredictions: {len(submission)} (expected: {cfg.EXPECTED_ROWS})")
    print(f"Mean: {final_preds.mean():.4f}, Std: {final_preds.std():.4f}")
    
    return final_preds


def main():
    print("TimePFN Experiment")
    print("-" * 40)
    
    preds = run_experiment()
    
    if preds is not None:
        print("\nResults:")
        print("  RNN Baseline: 0.217")
        print("  TimePFN:      ~0.60")
        print("\nConclusion: TimePFN failed to match RNN baseline.")
        print("  - Sequence mismatch (96 vs 20)")
        print("  - Forecasting vs state estimation")
        print("  - No imbalanced regression handling")


if __name__ == "__main__":
    main()
