"""Two-stage training utility for sideslip estimation."""

import torch
import torch.nn as nn
import numpy as np


def train_two_stage(model, train_loader, val_loader, band_dict, device,
                    stage1_epochs=30, stage2_epochs=40,
                    stage1_lr=3e-4, stage2_lr=5e-5,
                    threshold=2.0, slope=1.0, verbose=False):
    """Two-stage training: L1 then WeightedL1."""
    from utils.weighted_l1_loss import WeightedL1Loss
    from utils.band_mapping import calculate_band_mae
    
    model = model.to(device)
    
    # Stage 1: L1
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage1_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)
    
    for epoch in range(stage1_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
            optimizer.zero_grad()
            nn.L1Loss()(model(X), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    
    # Stage 2: WeightedL1
    loss_fn = WeightedL1Loss(threshold, slope)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    
    best_mae, best_state = float('inf'), None
    for epoch in range(stage2_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
            optimizer.zero_grad()
            loss_fn(model(X), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, val_loader, band_dict, device)
            if metrics['overall_mae'] < best_mae:
                best_mae = metrics['overall_mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    return model, evaluate(model, val_loader, band_dict, device)


@torch.no_grad()
def evaluate(model, loader, band_dict, device):
    """Evaluate model with band-wise metrics."""
    from utils.band_mapping import calculate_band_mae
    
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        preds.extend(model(X.to(device)).cpu().numpy().flatten())
        targets.extend(y.numpy())
    return calculate_band_mae(np.array(preds), np.array(targets), band_dict)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Get predictions and targets for visualization."""
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        preds.extend(model(X.to(device)).cpu().numpy().flatten())
        targets.extend(y.numpy())
    return np.array(preds), np.array(targets)
