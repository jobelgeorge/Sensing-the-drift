# ========== AUTOMATIC HYPERPARAMETER SWEEP FOR LAMBDA_FOCAL ==========
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.kaggle_metric import score

def train_with_lambda(lambda_focal_value, gamma_value, num_epochs, early_stop_thresh, 
                      train_loader, val_loader, model_class, model_params, 
                      criterion, lr, weight_decay, scheduler_params, device, seed=42,
                      use_scheduler=True):
    """
    Train a model with specific lambda_focal and return metrics.
    
    Returns:
        dict with keys: 'lambda_focal', 'best_val_loss', 'best_epoch', 
                       'many_score', 'medium_score', 'few_score', 'model_state'
    """
    from utils.reproducibility_utils import seed_everything
    
    # Reset seed for reproducibility
    seed_everything(seed)
    
    # Create fresh model
    model = model_class(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Optional scheduler
    scheduler = None
    if use_scheduler and scheduler_params is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    
    # Training loop
    best_loss = np.inf
    best_epoch = -1
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # ⚡ Setup AMP (Automatic Mixed Precision) for 2-3x speedup on GPU
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    print(f"\n{'='*80}")
    print(f"🔬 Config: λ={lambda_focal_value}, γ={gamma_value}, h={model_params.get('hidden_size')}, "
          f"fc={model_params.get('fc_units')}, layers={model_params.get('num_layers')}, "
          f"scheduler={'ON' if use_scheduler else 'OFF'}")
    print(f"⚡ AMP: {'ON' if use_amp else 'OFF (CPU)'}")
    print(f"{'='*80}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0
        
        for step, (x, y) in enumerate(train_loader):
            X, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).reshape([len(y), 1])
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # ⚡ Mixed precision forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred, hidden_seq = model(X)
                    base_loss = criterion(pred, y)
                    
                    # Focal L1 regularization (handle LSTM tuple output)
                    if isinstance(hidden_seq, tuple):
                        hidden_seq = hidden_seq[0]  # LSTM: (h_n, c_n) → use h_n
                    hidden_detached = hidden_seq.detach()
                    hidden_clamped = torch.clamp(hidden_detached, -10.0, 10.0)
                    reg_loss = (hidden_clamped.abs() ** gamma_value).mean()
                    train_loss = base_loss + lambda_focal_value * reg_loss
                
                running_loss += train_loss.item()
                
                # Scaled backward pass
                scaler.scale(train_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision (CPU)
                pred, hidden_seq = model(X)
                base_loss = criterion(pred, y)
                
                # Focal L1 regularization (handle LSTM tuple output)
                if isinstance(hidden_seq, tuple):
                    hidden_seq = hidden_seq[0]  # LSTM: (h_n, c_n) → use h_n
                hidden_detached = hidden_seq.detach()
                hidden_clamped = torch.clamp(hidden_detached, -10.0, 10.0)
                reg_loss = (hidden_clamped.abs() ** gamma_value).mean()
                train_loss = base_loss + lambda_focal_value * reg_loss
                
                running_loss += train_loss.item()
                
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        avg_train_loss = running_loss / (step + 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0
        
        with torch.no_grad():
            for val_step, (x, y) in enumerate(val_loader):
                X_val, y_val = x.to(device, non_blocking=True), y.to(device, non_blocking=True).reshape([len(y), 1])
                
                # ⚡ Use AMP for validation too
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        y_pred, _ = model(X_val)
                        val_loss = criterion(y_pred, y_val)
                else:
                    y_pred, _ = model(X_val)
                    val_loss = criterion(y_pred, y_val)
                
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / (val_step + 1)
        val_losses.append(avg_val_loss)
        
        # Update scheduler if enabled
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter = epoch - best_epoch
            if patience_counter >= early_stop_thresh:
                print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # ✅ Evaluate on VALIDATION SET for shot-based metrics (not training!)
    # This prevents overfitting and gives true generalization performance
    model.eval()
    preds = []
    ground_truths = []
    
    # Use validation loader (already unshuffled)
    with torch.no_grad():
        for x, y in val_loader:
            X, y = x.to(device), y.to(device).reshape([len(y), 1])
            pred, _ = model(X)
            preds.append(pred.cpu().numpy().flatten())
            ground_truths.append(y.cpu().numpy().flatten())
    
    preds = np.concatenate(preds)
    ground_truths = np.concatenate(ground_truths)
    
    # Calculate shot-based scores on validation data
    from utils.band_mapping import load_band_mapping
    shot_band_mapping = load_band_mapping()
    
    gt_df = pd.DataFrame({"ground_truth_sideslip": ground_truths}, dtype=np.float64)
    gt_df["id"] = range(len(ground_truths))
    gt_df["band"] = gt_df["ground_truth_sideslip"].round(1).map(shot_band_mapping).fillna("unknown")
    
    preds_df = pd.DataFrame({"id": range(len(preds)), "predicted_sideslip": preds})
    
    shot_scores = score(gt_df, preds_df, row_id_column_name="id")
    
    print(f"  ✓ Best Val Loss: {best_loss:.5f} (epoch {best_epoch+1})")
    print(f"  ✓ Validation Scores:")
    print(f"     Many-Shot: {shot_scores['many']:.3f}°")
    print(f"     Medium-Shot: {shot_scores['medium']:.3f}°")
    print(f"     Few-Shot: {shot_scores['few']:.3f}°")
    
    return {
        'lambda_focal': lambda_focal_value,
        'gamma': gamma_value,
        'hidden_size': model_params.get('hidden_size'),
        'fc_units': model_params.get('fc_units'),
        'num_layers': model_params.get('num_layers'),
        'rnn_type': model_params.get('rnn_type', 'RNN'),  # 🆕 Added rnn_type
        'use_scheduler': use_scheduler,
        'best_val_loss': best_loss,
        'best_epoch': best_epoch,
        'many_score': shot_scores['many'],
        'medium_score': shot_scores['medium'],
        'few_score': shot_scores['few'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_state': best_model_state
    }


def compare_lambda_focals(lambda_values, gamma_value, num_epochs, early_stop_thresh,
                          train_loader, val_loader, model_class, model_params,
                          criterion, lr, weight_decay, scheduler_params, device,
                          use_scheduler_options=[True]):
    """
    Compare multiple lambda_focal values and visualize results.
    
    Args:
        lambda_values: list of lambda_focal values to test
        gamma_value: gamma parameter for focal L1
        use_scheduler_options: list of boolean values [True, False] to test with/without scheduler
        Other parameters: same as train_with_lambda
    
    Returns:
        pd.DataFrame with comparison results
    """
    results = []
    
    print("\n" + "="*80)
    print("🚀 STARTING HYPERPARAMETER SWEEP")
    print("="*80)
    print(f"Testing {len(lambda_values)} different lambda_focal values")
    print(f"Lambda values: {lambda_values}")
    print(f"Gamma: {gamma_value}")
    print(f"Scheduler options: {use_scheduler_options}")
    
    experiment_idx = 0
    for use_sched in use_scheduler_options:
        for lambda_val in lambda_values:
            print(f"\n[{experiment_idx+1}/{len(lambda_values)*len(use_scheduler_options)}] "
                  f"Testing lambda_focal={lambda_val}, scheduler={'ON' if use_sched else 'OFF'}")
            
            result = train_with_lambda(
                lambda_focal_value=lambda_val,
                gamma_value=gamma_value,
                num_epochs=num_epochs,
                early_stop_thresh=early_stop_thresh,
                train_loader=train_loader,
                val_loader=val_loader,
                model_class=model_class,
                model_params=model_params,
                criterion=criterion,
                lr=lr,
                weight_decay=weight_decay,
                scheduler_params=scheduler_params,
                device=device,
                seed=42,  # FIXED SEED for fair comparison
                use_scheduler=use_sched
            )
            
            results.append(result)
            experiment_idx += 1
    
    # Create comparison DataFrame
    df_results = pd.DataFrame([
        {
            'lambda_focal': r['lambda_focal'],
            'gamma': r['gamma'],
            'hidden_size': r['hidden_size'],
            'fc_units': r['fc_units'],
            'num_layers': r['num_layers'],
            'scheduler': 'YES' if r['use_scheduler'] else 'NO',
            'best_val_loss': r['best_val_loss'],
            'best_epoch': r['best_epoch'],
            'many': r['many_score'],
            'medium': r['medium_score'],
            'few': r['few_score']
        }
        for r in results
    ])
    
    # Display summary
    print("\n" + "="*80)
    print("📊 HYPERPARAMETER SWEEP RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Find best configurations
    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATIONS")
    print("="*80)
    
    best_val_idx = df_results['best_val_loss'].idxmin()
    best_many_idx = df_results['many'].idxmin()
    best_overall_idx = (df_results['many'] + df_results['medium'] + df_results['few']).idxmin()
    
    print(f"\n✅ Best Validation Loss:")
    print(f"   lambda_focal={df_results.loc[best_val_idx, 'lambda_focal']}")
    print(f"   Val Loss: {df_results.loc[best_val_idx, 'best_val_loss']:.5f}")
    
    print(f"\n✅ Best Many-Shot Score:")
    print(f"   lambda_focal={df_results.loc[best_many_idx, 'lambda_focal']}")
    print(f"   Many: {df_results.loc[best_many_idx, 'many']:.3f}°")
    
    print(f"\n✅ Best Overall (sum of all shots):")
    print(f"   lambda_focal={df_results.loc[best_overall_idx, 'lambda_focal']}")
    print(f"   Many: {df_results.loc[best_overall_idx, 'many']:.3f}°")
    print(f"   Medium: {df_results.loc[best_overall_idx, 'medium']:.3f}°")
    print(f"   Few: {df_results.loc[best_overall_idx, 'few']:.3f}°")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Validation Loss vs Lambda
    ax = axes[0, 0]
    ax.plot(df_results['lambda_focal'], df_results['best_val_loss'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('lambda_focal', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Validation Loss vs Lambda Focal', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df_results['best_val_loss'].min(), color='red', linestyle='--', alpha=0.5, label='Best')
    ax.legend()
    
    # Plot 2: Shot-based scores vs Lambda
    ax = axes[0, 1]
    ax.plot(df_results['lambda_focal'], df_results['many'], 'o-', label='Many-Shot', linewidth=2, markersize=8)
    ax.plot(df_results['lambda_focal'], df_results['medium'], 's-', label='Medium-Shot', linewidth=2, markersize=8)
    ax.plot(df_results['lambda_focal'], df_results['few'], '^-', label='Few-Shot', linewidth=2, markersize=8)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Kaggle Limit (Many)')
    ax.set_xlabel('lambda_focal', fontsize=12)
    ax.set_ylabel('MAE (degrees)', fontsize=12)
    ax.set_title('Shot-based Scores vs Lambda Focal', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training curves for best model
    best_result = results[best_overall_idx]
    ax = axes[1, 0]
    ax.plot(best_result['train_losses'], label='Train Loss', linewidth=2)
    ax.plot(best_result['val_losses'], label='Val Loss', linewidth=2)
    ax.axvline(x=best_result['best_epoch'], color='gray', linestyle='--', alpha=0.7, label='Best Epoch')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Best Model Training Curve (λ={best_result["lambda_focal"]})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Comparison table as text
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    table_data.append(['Lambda', 'Val Loss', 'Many', 'Medium', 'Few'])
    for _, row in df_results.iterrows():
        table_data.append([
            f"{row['lambda_focal']:.5f}",
            f"{row['best_val_loss']:.5f}",
            f"{row['many']:.3f}",
            f"{row['medium']:.3f}",
            f"{row['few']:.3f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Highlight header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    best_val_row = df_results['best_val_loss'].idxmin() + 1
    best_many_row = df_results['many'].idxmin() + 1
    
    table[(best_val_row, 1)].set_facecolor('#90EE90')
    table[(best_many_row, 2)].set_facecolor('#90EE90')
    
    ax.set_title('Results Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return df_results, results


# Usage example (to be placed in a new notebook cell):
"""
# Define lambda values to test
lambda_values_to_test = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]

# Run comparison
df_results, all_results = compare_lambda_focals(
    lambda_values=lambda_values_to_test,
    gamma_value=gamma,
    num_epochs=num_epochs,
    early_stop_thresh=early_stop_thresh,
    train_loader=train_loader,
    val_loader=val_loader,
    model_class=RNN,
    model_params={
        'input_size': X_train.shape[1],
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'fc_units': fc_units,
        'dropout': dropouts,
        'rnn_type': rnn_type,
        'output_size': 1
    },
    criterion=criterion,
    lr=lr,
    weight_decay=weight_decay,
    scheduler_params={
        'mode': scheduler_mode,
        'factor': scheduler_factor,
        'patience': scheduler_patience,
        'min_lr': scheduler_min_lr
    },
    device=device
)
"""
