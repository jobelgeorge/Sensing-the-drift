"""
Production Training Script V5 - Optimized for Kaggle Competition
================================================================

Key Features:
✅ Optimal validation files [9, 1, 6] - scientifically chosen for quality
✅ Official band mapping from sideSlip_distribution.csv
✅ Lower batch size (32) for better generalization  
✅ Minimal dropout (0.2) for production stability
✅ Advanced loss: Focal-R Loss (handles imbalanced regression)
✅ Proper checkpointing with best model saving
✅ Early stopping to prevent overfitting
✅ Comprehensive logging and metrics tracking

Competition Goal:
- GATE: Many-shot MAE ≤ 0.20° (must pass!)
- SCORE: Few-shot MAE (minimize for leaderboard)

Model: BiLSTM with 67k parameters
Dataset: Uses Files [9, 1, 6] for validation (optimal quality + representativeness)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.rnn import RNNModel  # ✅ Using simple RNN
from utils.data_loadingV4 import load_data
from utils.dir_lds_loss import DIR_LDS_Loss  # ✅ Official DIR implementation
from utils.simple_losses import FocalRLoss, SimpleLDSLoss  # ✅ Simple Focal-R for comparison
from utils.reproducibility_utils import seed_everything  # ✅ Fixed function name
from utils.band_mapping import load_band_mapping, calculate_band_mae


# ==================== CONFIGURATION ====================
class Config:
    """Production training configuration"""
    
    # Data
    TRAIN_PATH = r"datasets_challenge\train"
    VAL_FILES = [0, 5, 7]  # train_part_001 (11.8k), train_part_006 (20.5k), train_part_008 (22.2k) = 54.5k samples
    SEQUENCE_LENGTH = 20
    TARGET = "sideSlip"
    THRESHOLD = True
    THRESHOLD_VALUE = 3.0  # Remove low-speed noise
    THRESHOLD_COLUMN = "vxCG"
    
    # Model Architecture
    INPUT_SIZE = 22  # Number of features
    HIDDEN_SIZE = 128  
    NUM_LAYERS = 3  # Increased from 2 → deeper (extract better features)
    DROPOUT = 0.0  # Reduced from 0.2 → less noise (more stable training)
    
    # Training Hyperparameters
    BATCH_SIZE = 64  # Increased from 32 → 2x larger (stable gradients!)
    LEARNING_RATE = 3e-5  # Reduced from 1e-4 → 3.3x slower (prevent oscillations)
    WEIGHT_DECAY = 1e-4  # L2 regularization
    NUM_EPOCHS = 100
    GRADIENT_CLIP = 1.0
    
    # Loss Configuration - EXPERIMENT WITH THIS!
    LOSS_TYPE = "DIR_LDS"  # Options: "DIR_LDS", "FOCAL_R", "SIMPLE_LDS"  ← START HERE!
    
    # DIR-LDS Loss hyperparameters
    LDS_BASE = 'smoothl1'        # Base loss: 'l1', 'smoothl1', 'mse'
    LDS_SMOOTHL1_BETA = 0.1      # Huber-like transition
    LDS_REWEIGHT = 'sqrt_inv'    # Reweighting: 'inv' or 'sqrt_inv'
    LDS_SIGMA = 2.0              # Smoothing bandwidth
    LDS_NUM_BINS = 1000          # Discretization bins
    
    # Competition Target (from Kaggle)
    TARGET_MANY_MAE = 0.20  # Must be ≤ 0.20° to pass gate!
    
    # Band Distribution (official mapping)
    BAND_DISTRIBUTION_PATH = "datasets_challenge/sideSlip_distribution.csv"
    
    # Early Stopping
    EARLY_STOP_PATIENCE = 15
    MIN_DELTA = 1e-5
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    
    # Reproducibility
    SEED = 42
    DETERMINISTIC = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __repr__(self):
        return "\n".join(f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))


# ==================== METRICS CALCULATION ====================
def calculate_metrics(model, dataloader, device, band_dict):
    """Calculate comprehensive metrics using official band mapping"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            outputs = model(X)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    predictions = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Use official band mapping from distribution file
    metrics = calculate_band_mae(predictions, targets, band_dict)
    
    return metrics


# ==================== EARLY STOPPING ====================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=1e-5, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ==================== TRAINING LOOP ====================
def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y.unsqueeze(1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


# ==================== MAIN TRAINING FUNCTION ====================
def train():
    """Main training function"""
    
    # Configuration
    config = Config()
    
    print("\n" + "="*80)
    print("🚀 PRODUCTION TRAINING V5 - Kaggle Competition")
    print("="*80)
    print("\n📋 Configuration:")
    print(config)
    print("\n" + "="*80)
    
    # Set seed
    seed_everything(config.SEED)
    print(f"\n✅ Seed set to {config.SEED} (deterministic={config.DETERMINISTIC})")
    
    # Load band mapping
    print(f"\n📊 Loading official band mapping from {config.BAND_DISTRIBUTION_PATH}...")
    band_dict = load_band_mapping(config.BAND_DISTRIBUTION_PATH)
    print(f"✅ Loaded {len(band_dict)} unique sideslip values with band classifications")
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.CHECKPOINT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"✅ Checkpoint directory: {run_dir}")
    
    # Save configuration
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=4, default=str)
    print(f"✅ Configuration saved to {config_path}")
    
    # Load data
    print("\n" + "="*80)
    print("📂 LOADING DATA")
    print("="*80)
    
    train_loader, val_loader, pipeline, mean_dict, std_dict = load_data(
        train_path=config.TRAIN_PATH,
        val_files=config.VAL_FILES,  # Optimal files!
        columns_to_drop=['sideSlip', 'vyCG', 'longitude', 'latitude', 'time'],  # Drop target + unused features
        sequence_length=config.SEQUENCE_LENGTH,
        target=config.TARGET,
        threshold=config.THRESHOLD,
        threshold_value=config.THRESHOLD_VALUE,
        threshold_column=config.THRESHOLD_COLUMN,
        batchsize=config.BATCH_SIZE,
        scaling=True,
        seed=config.SEED,
        shuffle=True
    )
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Collect training targets for DIR-LDS Loss
    print("\n" + "="*80)
    print("📊 COLLECTING TRAINING TARGETS FOR LDS")
    print("="*80)
    print("Iterating through training data to collect all target values...")
    
    train_targets = []
    for batch in train_loader:
        _, y = batch
        train_targets.append(y.numpy())
    train_targets = np.concatenate(train_targets).flatten()
    
    print(f"✅ Collected {len(train_targets):,} training targets")
    print(f"   Range: [{train_targets.min():.2f}, {train_targets.max():.2f}]")
    print(f"   Mean: {train_targets.mean():.2f}, Std: {train_targets.std():.2f}")
    
    # Initialize model
    print("\n" + "="*80)
    print("🏗️  BUILDING MODEL")
    print("="*80)
    
    model = RNNModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_batch_norm=False,
        output_size=1
    ).to(config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model initialized: RNN")
    print(f"   Parameters: {num_params:,}")
    print(f"   Device: {config.DEVICE}")
    print(f"   Hidden size: {config.HIDDEN_SIZE}")
    print(f"   Layers: {config.NUM_LAYERS}")
    print(f"   Dropout: {config.DROPOUT}")
    
    # Initialize loss function based on config
    print("\n" + "="*80)
    print(f"🎯 INITIALIZING LOSS FUNCTION: {config.LOSS_TYPE}")
    print("="*80)
    
    if config.LOSS_TYPE == "DIR_LDS":
        # Official DIR-LDS with Gaussian smoothing
        criterion = DIR_LDS_Loss(
            train_targets=train_targets,
            base=config.LDS_BASE,
            smoothl1_beta=config.LDS_SMOOTHL1_BETA,
            reweight=config.LDS_REWEIGHT,
            lds=True,  # Enable smoothing
            lds_kernel='gaussian',
            lds_ks=5,
            lds_sigma=config.LDS_SIGMA,
            num_bins=config.LDS_NUM_BINS
        )
    
    elif config.LOSS_TYPE == "FOCAL_R":
        # Simple Focal-R Loss (focuses on hard examples)
        criterion = FocalRLoss(
            gamma=1.0,  # Moderate focusing
            beta=config.LDS_SMOOTHL1_BETA,
            base=config.LDS_BASE
        )
    
    elif config.LOSS_TYPE == "SIMPLE_LDS":
        # LDS without Gaussian smoothing (baseline)
        criterion = SimpleLDSLoss(
            train_targets=train_targets,
            base=config.LDS_BASE,
            beta=config.LDS_SMOOTHL1_BETA,
            num_bins=config.LDS_NUM_BINS
        )
    
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {config.LOSS_TYPE}")
    
    print(f"\n✅ Loss function: {criterion}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning Rate Scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize validation loss
        factor=0.5,  # Halve learning rate
        patience=5,  # After 5 epochs without improvement
        verbose=True,  # Print when LR changes
        min_lr=1e-6  # Don't go below this
    )
    
    print(f"\n✅ Optimizer: Adam (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    print(f"✅ LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        min_delta=config.MIN_DELTA,
        mode='min'
    )
    
    # Training loop
    print("\n" + "="*80)
    print("🏋️  TRAINING")
    print("="*80)
    print(f"\nTarget: Many-shot MAE ≤ {config.TARGET_MANY_MAE}° (MUST PASS!)\n")
    
    best_val_loss = float('inf')
    best_many_mae = float('inf')
    best_epoch = 0
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_overall_mae': [],
        'val_many_mae': [],
        'val_few_mae': []
    }
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, config)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, config.DEVICE)
        
        # Calculate metrics
        metrics = calculate_metrics(model, val_loader, config.DEVICE, band_dict)
        
        # Update history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_overall_mae'].append(metrics['overall_mae'])
        training_history['val_many_mae'].append(metrics['many_mae'])
        training_history['val_few_mae'].append(metrics['few_mae'])
        
        # Status indicator
        status = ""
        if metrics['many_mae'] <= config.TARGET_MANY_MAE:
            status = "✅ PASS"
        else:
            status = "⚠️  GATE"
        
        # Print progress
        print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Many: {metrics['many_mae']:.4f}° {status} | "
              f"Few: {metrics['few_mae']:.4f}°")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_many_mae = metrics['many_mae']
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
                'config': vars(config)
            }
            
            best_path = os.path.join(run_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model (Val Loss: {val_loss:.4f}, Many MAE: {metrics['many_mae']:.4f}°)")
        
        # Periodic checkpoint
        if epoch % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, checkpoint_path)
        
        # Learning rate scheduler step (reduce LR on plateau)
        scheduler.step(val_loss)
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n⏸️  Early stopping triggered at epoch {epoch}")
            print(f"   Best epoch: {best_epoch}")
            print(f"   Best val loss: {best_val_loss:.4f}")
            print(f"   Best many MAE: {best_many_mae:.4f}°")
            break
    
    # Save training history
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"\n✅ Training history saved to {history_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 TRAINING COMPLETE")
    print("="*80)
    print(f"\n📊 Best Results (Epoch {best_epoch}):")
    print(f"   Validation Loss: {best_val_loss:.4f}")
    print(f"   Many-shot MAE: {best_many_mae:.4f}° {'✅ PASS' if best_many_mae <= config.TARGET_MANY_MAE else '⚠️  FAIL'}")
    print(f"\n💾 Best model saved to: {best_path}")
    print(f"📁 Run directory: {run_dir}")
    print("\n" + "="*80 + "\n")
    
    return model, training_history, run_dir


if __name__ == "__main__":
    try:
        model, history, run_dir = train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
