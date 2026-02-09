"""
Chronos Analysis for Sideslip Estimation
Baseline: 0.217 | Chronos Best: 0.543
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Path setup - Find project root by checking for utils/ and models/ folders
def find_project_root():
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "utils").exists() and (p / "models").exists():
            return p
    return here

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project imports
from utils.data_loading import load_data, feature_names
from utils.reproducibility_utils import seed_everything
from utils.visualization import plot_method_comparison


class Config:
    TRAIN_PATH = PROJECT_ROOT / "datasets_challenge" / "train"
    SEQ_LENGTH = 20
    BATCH_SIZE = 256
    TARGET = 'sideSlip'
    COLUMNS_TO_DROP = ["sideSlip", "vyCG", "time", "longitude", "latitude"]
    TRAIN_RATIO = 0.8
    SEED = 42
    CHRONOS_MODEL = "amazon/chronos-t5-small"
    KEY_SENSORS = ['yawRate', 'ayCG', 'handwheelAngle', 'rollRate', 'vxCG']
    MLP_HIDDEN = 256
    MLP_EPOCHS = 30
    MLP_LR = 1e-3


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_chronos_features(chronos_pipeline, x_batch, sensor_indices, device):
    """Extract Chronos embeddings for batch."""
    embeddings = []
    for idx in sensor_indices:
        sensor_data = x_batch[:, :, idx].cpu()
        with torch.no_grad():
            tok_out = chronos_pipeline.tokenizer.context_input_transform(sensor_data)
            token_ids = tok_out[0].to(device)
            attn_mask = tok_out[1].to(device)
            enc_out = chronos_pipeline.model.model.encoder(
                input_ids=token_ids, attention_mask=attn_mask, return_dict=True
            )
            hidden = enc_out.last_hidden_state
            mask_exp = attn_mask.unsqueeze(-1).float()
            embed = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            embeddings.append(embed.cpu().numpy())
    return np.concatenate(embeddings, axis=1)


def run_zero_shot(chronos, train_loader, val_loader, sensor_indices, device):
    """Zero-shot: Chronos embeddings + Ridge regression."""
    from sklearn.linear_model import Ridge
    
    print("Extracting train embeddings...")
    X_train, y_train = [], []
    for bx, by in tqdm(train_loader):
        bx = bx.to(device)
        feat = extract_chronos_features(chronos, bx, sensor_indices, device)
        raw = bx[:, -1, :].cpu().numpy()
        X_train.append(np.hstack([feat, raw]))
        y_train.append(by.numpy())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    print("Extracting val embeddings...")
    X_val, y_val = [], []
    for bx, by in tqdm(val_loader):
        bx = bx.to(device)
        feat = extract_chronos_features(chronos, bx, sensor_indices, device)
        raw = bx[:, -1, :].cpu().numpy()
        X_val.append(np.hstack([feat, raw]))
        y_val.append(by.numpy())
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    mae = np.mean(np.abs(ridge.predict(X_val) - y_val))
    
    print(f"Zero-Shot MAE: {mae:.4f} (Baseline: 0.217)")
    return X_train, y_train, X_val, y_val, mae


def run_finetuned(X_train, y_train, X_val, y_val, config, device):
    """Fine-tuned: MLP head on Chronos embeddings."""
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)
    
    model = MLPHead(X_train.shape[1], config.MLP_HIDDEN).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.MLP_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MLP_EPOCHS)
    
    best_mae = float('inf')
    print(f"Training MLP ({sum(p.numel() for p in model.parameters()):,} params)...")
    
    for epoch in range(config.MLP_EPOCHS):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in val_dl:
                preds.extend(model(bx.to(device)).cpu().numpy())
                targets.extend(by.numpy())
        mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
        best_mae = min(best_mae, mae)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: MAE={mae:.4f}")
    
    print(f"Fine-Tuned MAE: {best_mae:.4f} (Baseline: 0.217)")
    return best_mae


def main():
    config = Config()
    seed_everything(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Model: {config.CHRONOS_MODEL}")
    
    try:
        from chronos import ChronosPipeline
        chronos = ChronosPipeline.from_pretrained(
            config.CHRONOS_MODEL, device_map=device, torch_dtype=torch.float32
        )
    except ImportError:
        print("Install: pip install chronos-forecasting")
        return
    
    train_loader, val_loader, _ = load_data(
        train_path=str(config.TRAIN_PATH),
        train_ratio=config.TRAIN_RATIO,
        columns_to_standardize=feature_names,
        columns_to_drop=config.COLUMNS_TO_DROP,
        sequence_length=config.SEQ_LENGTH,
        target=config.TARGET,
        threshold=True, threshold_value=3.0, threshold_column="vxCG",
        batchsize=config.BATCH_SIZE, scaling=True, seed=config.SEED, shuffle=True
    )
    
    sensor_indices = [feature_names.index(n) for n in config.KEY_SENSORS if n in feature_names]
    
    X_train, y_train, X_val, y_val, zs_mae = run_zero_shot(
        chronos, train_loader, val_loader, sensor_indices, device
    )
    ft_mae = run_finetuned(X_train, y_train, X_val, y_val, config, device)
    
    print("\nResults:")
    print(f"  RNN Baseline:    0.217")
    print(f"  Chronos ZS:      {zs_mae:.3f}")
    print(f"  Chronos FT:      {ft_mae:.3f}")
    print("\nConclusion: Chronos failed to match RNN baseline.")


if __name__ == "__main__":
    main()
