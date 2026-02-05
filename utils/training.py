"""
Two-Stage Training Utilities for Sideslip Estimation
=====================================================

Core training functions for the winning two-stage approach:
- Stage 1: L1 warm-up (general feature learning)
- Stage 2: WeightedL1 refinement (tail optimization)

Best configuration: S1=30 epochs, S2=40 epochs, τ=2.0, s=1.0
Kaggle Score: 0.21751
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def train_two_stage(model, train_loader, val_loader, band_dict, device,
                    stage1_epochs=30, stage2_epochs=40,
                    stage1_lr=3e-4, stage2_lr=5e-5,
                    threshold=2.0, slope=1.0, 
                    verbose=False, return_history=False):
    """
    Two-stage training: L1 warm-up then WeightedL1 refinement.
    
    Args:
        model: PyTorch model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        band_dict: Band mapping dictionary for metrics
        device: 'cuda' or 'cpu'
        stage1_epochs: Number of Stage 1 epochs (default: 30)
        stage2_epochs: Number of Stage 2 epochs (default: 40)
        stage1_lr: Stage 1 learning rate (default: 3e-4)
        stage2_lr: Stage 2 learning rate (default: 5e-5)
        threshold: WeightedL1 threshold τ (default: 2.0)
        slope: WeightedL1 slope s (default: 1.0)
        verbose: Print progress (default: False)
        return_history: Return training history (default: False)
    
    Returns:
        model: Trained model
        metrics: Final evaluation metrics
        history: (optional) Training history dict if return_history=True
    
    Example:
        >>> from models.vanilla_rnn import VanillaRNN
        >>> model = VanillaRNN(input_size=22)
        >>> model, metrics, history = train_two_stage(
        ...     model, train_loader, val_loader, band_dict, 'cuda',
        ...     return_history=True
        ... )
    """
    from utils.weighted_l1_loss import WeightedL1Loss
    from utils.band_mapping import calculate_band_mae
    
    model = model.to(device)
    history = {
        'stage1_train': [], 'stage1_val': [],
        'stage2_train': [], 'stage2_val': []
    }
    
    # ==================== Stage 1: L1 Loss ====================
    if verbose:
        print(f"Stage 1: L1 warm-up ({stage1_epochs} epochs, lr={stage1_lr})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage1_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)
    
    for epoch in range(stage1_epochs):
        # Training
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
            optimizer.zero_grad()
            loss = nn.L1Loss()(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
                val_losses.append(nn.L1Loss()(model(X), y).item())
        
        history['stage1_train'].append(np.mean(train_losses))
        history['stage1_val'].append(np.mean(val_losses))
        scheduler.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{stage1_epochs}: train={history['stage1_train'][-1]:.4f}, val={history['stage1_val'][-1]:.4f}")
    
    # ==================== Stage 2: WeightedL1 ====================
    if verbose:
        print(f"Stage 2: WeightedL1 refinement ({stage2_epochs} epochs, lr={stage2_lr}, τ={threshold}, s={slope})")
    
    loss_fn = WeightedL1Loss(threshold, slope)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    
    best_mae, best_state = float('inf'), None
    
    for epoch in range(stage2_epochs):
        # Training
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation (use L1 for consistent comparison)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
                val_losses.append(nn.L1Loss()(model(X), y).item())
        
        val_loss = np.mean(val_losses)
        history['stage2_train'].append(np.mean(train_losses))
        history['stage2_val'].append(val_loss)
        
        # Track best model
        if val_loss < best_mae:
            best_mae = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{stage2_epochs}: train={history['stage2_train'][-1]:.4f}, val={val_loss:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    final_metrics = evaluate(model, val_loader, band_dict, device)
    
    if verbose:
        print(f"Training complete. Best val MAE: {best_mae:.4f}")
    
    if return_history:
        return model, final_metrics, history
    return model, final_metrics


def train_single_stage(model, train_loader, val_loader, loss_fn,
                       epochs=70, lr=3e-4, device='cuda',
                       verbose=False, return_history=False):
    """
    Single-stage training with specified loss function.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        loss_fn: Loss function instance
        epochs: Number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        verbose: Print progress
        return_history: Return training history
    
    Returns:
        model: Trained model
        history: (optional) Training history if return_history=True
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_loss': []}
    best_loss, best_state = float('inf'), None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1) if y.dim() == 1 else y.to(device)
                val_losses.append(nn.L1Loss()(model(X), y).item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    if return_history:
        return model, history
    return model


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
