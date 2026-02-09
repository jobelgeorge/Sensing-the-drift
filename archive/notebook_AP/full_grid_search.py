# ========== COMPREHENSIVE GRID SEARCH FOR MODEL HYPERPARAMETERS ==========
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.kaggle_metric import score
from hyperparameter_sweep import train_with_lambda
import itertools
import gc  # Garbage collection for memory optimization
import sys
from pathlib import Path

# Add parent directory to path to import alternative loss functions
sys.path.append(str(Path(__file__).parent))

# Import alternative loss functions from the dedicated file
from alternative_loss_functions import (
    HuberLoss, 
    CombinedLoss, 
    QuantileLoss,
    WeightedMSELoss,
    ImprovedFocalL1Loss
)


def comprehensive_grid_search(
    # Training parameters (required)
    num_epochs,
    early_stop_thresh,
    train_loader,
    val_loader,
    model_class,
    criterion,
    lr,
    weight_decay,
    scheduler_params,
    device,
    input_size,
    dropout,
    
    # Grid search parameters (required)
    hidden_sizes,
    fc_units_list,
    num_layers_list,
    use_scheduler_options,
    
    # Optional legacy parameters (for backward compatibility)
    lambda_focal_values=None,
    gamma_values=None,
    
    # Optional parameters with defaults
    rnn_types=['RNN'],  # 🆕 List of RNN types to test
    loss_function_configs=None,  # 🆕 List of loss function configs to test
    max_experiments=None,  # Limit number of experiments
    save_every_n=10  # Save intermediate results every N experiments
):
    """
    Comprehensive grid search over all model hyperparameters.
    
    Tests all combinations of:
    - hidden_size values
    - fc_units values
    - num_layers values
    - scheduler on/off
    - RNN types (RNN, LSTM, GRU, etc.)
    - Loss functions (L1, MSE, Huber, etc.)
    
    Args:
        lambda_focal_values: DEPRECATED - kept for backward compatibility
        gamma_values: DEPRECATED - kept for backward compatibility
        hidden_sizes: list of hidden sizes (e.g., [32, 64])
        fc_units_list: list of FC units (e.g., [32, 64])
        num_layers_list: list of num layers (e.g., [1, 2])
        use_scheduler_options: list of booleans (e.g., [True, False])
        loss_function_configs: list of dicts with loss config (e.g., [{'name': 'L1', 'loss': nn.L1Loss()}])
        max_experiments: if set, limits total experiments (uses random sampling)
        save_every_n: saves intermediate results every N experiments
    
    Returns:
        pd.DataFrame with all results
    """
    
    # Default loss functions if not specified
    if loss_function_configs is None:
        loss_function_configs = [
            {'name': 'L1', 'loss': nn.L1Loss()},
        ]
    
    # Set defaults for legacy parameters (backward compatibility)
    if lambda_focal_values is None:
        lambda_focal_values = [0.0]  # No regularization by default
    if gamma_values is None:
        gamma_values = [0.5]  # Default gamma
    
    # Generate all combinations
    all_combinations = list(itertools.product(
        lambda_focal_values,
        gamma_values,
        hidden_sizes,
        fc_units_list,
        num_layers_list,
        use_scheduler_options,
        rnn_types,  # 🆕 Include RNN type in grid
        loss_function_configs  # 🆕 Include loss function in grid
    ))
    
    total_experiments = len(all_combinations)
    
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE GRID SEARCH")
    print("="*80)
    print(f"Total possible combinations: {total_experiments}")
    print(f"Hidden size: {len(hidden_sizes)} values {hidden_sizes}")
    print(f"FC units: {len(fc_units_list)} values {fc_units_list}")
    print(f"Num layers: {len(num_layers_list)} values {num_layers_list}")
    print(f"Scheduler: {len(use_scheduler_options)} options")
    print(f"RNN types: {len(rnn_types)} architectures {rnn_types}")
    print(f"Loss functions: {len(loss_function_configs)} types {[lf['name'] for lf in loss_function_configs]}")
    if lambda_focal_values != [0.0]:
        print(f"⚠️  Lambda focal: {len(lambda_focal_values)} values (legacy parameter)")
    if gamma_values != [0.5]:
        print(f"⚠️  Gamma: {len(gamma_values)} values (legacy parameter)")
    
    # Limit experiments if requested
    if max_experiments and total_experiments > max_experiments:
        print(f"\n⚠️  Limiting to {max_experiments} random experiments (from {total_experiments} total)")
        np.random.seed(42)
        indices = np.random.choice(total_experiments, max_experiments, replace=False)
        all_combinations = [all_combinations[i] for i in sorted(indices)]
        total_experiments = max_experiments
    
    print(f"\n🚀 Running {total_experiments} experiments...")
    
    results = []
    
    for exp_idx, (lambda_val, gamma_val, h_size, fc_size, n_layers, use_sched, rnn_type, loss_config) in enumerate(all_combinations):
        print(f"\n{'='*80}")
        print(f"[{exp_idx+1}/{total_experiments}] Experiment Configuration:")
        print(f"  λ_focal={lambda_val}, γ={gamma_val}, hidden={h_size}, fc={fc_size}, "
              f"layers={n_layers}, rnn_type={rnn_type}, loss={loss_config['name']}, scheduler={'ON' if use_sched else 'OFF'}")
        print(f"{'='*80}")
        
        # Create model parameters dict
        model_params = {
            'input_size': input_size,
            'hidden_size': h_size,
            'num_layers': n_layers,
            'fc_units': fc_size,
            'dropout': dropout,
            'rnn_type': rnn_type,
            'output_size': 1
        }
        
        try:
            # Train model with these parameters
            result = train_with_lambda(
                lambda_focal_value=lambda_val,
                gamma_value=gamma_val,
                num_epochs=num_epochs,
                early_stop_thresh=early_stop_thresh,
                train_loader=train_loader,
                val_loader=val_loader,
                model_class=model_class,
                model_params=model_params,
                criterion=loss_config['loss'],  # 🆕 Use loss from config
                lr=lr,
                weight_decay=weight_decay,
                scheduler_params=scheduler_params,
                device=device,
                seed=42,  # FIXED SEED for fair comparison across all experiments
                use_scheduler=use_sched
            )
            
            # Add loss function name to result
            result['loss_function'] = loss_config['name']
            
            # Don't save model yet - will save only best models at the end
            results.append(result)
            
            # ⚡ Memory cleanup after each experiment
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save intermediate results
            if (exp_idx + 1) % save_every_n == 0:
                df_temp = pd.DataFrame([
                    {
                        'lambda_focal': r['lambda_focal'],
                        'gamma': r['gamma'],
                        'hidden_size': r['hidden_size'],
                        'fc_units': r['fc_units'],
                        'num_layers': r['num_layers'],
                        'rnn_type': r['rnn_type'],
                        'loss_function': r.get('loss_function', 'L1'),  # 🆕
                        'scheduler': 'YES' if r['use_scheduler'] else 'NO',
                        'best_val_loss': r['best_val_loss'],
                        'best_epoch': r['best_epoch'],
                        'many': r['many_score'],
                        'medium': r['medium_score'],
                        'few': r['few_score'],
                        'total_score': r['many_score'] + r['medium_score'] + r['few_score']
                    }
                    for r in results
                ])
                df_temp.to_csv('grid_search_intermediate.csv', index=False)
                print(f"\n💾 Saved intermediate results ({len(results)} experiments so far)")
        
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            continue
    
    # Create final DataFrame
    df_results = pd.DataFrame([
        {
            'lambda_focal': r['lambda_focal'],
            'gamma': r['gamma'],
            'hidden_size': r['hidden_size'],
            'fc_units': r['fc_units'],
            'num_layers': r['num_layers'],
            'rnn_type': r['rnn_type'],
            'loss_function': r.get('loss_function', 'L1'),  # 🆕
            'scheduler': 'YES' if r['use_scheduler'] else 'NO',
            'best_val_loss': r['best_val_loss'],
            'best_epoch': r['best_epoch'],
            'many': r['many_score'],
            'medium': r['medium_score'],
            'few': r['few_score'],
            'total_score': r['many_score'] + r['medium_score'] + r['few_score']
        }
        for r in results
    ])
    
    # Save final results
    df_results.to_csv('grid_search_final.csv', index=False)
    
    # ========== SAVE ONLY BEST MODELS ==========
    print("\n" + "="*80)
    print("💾 SAVING BEST MODELS")
    print("="*80)
    
    # Find indices of best models
    best_total_idx = df_results['total_score'].idxmin()
    best_few_idx = df_results['few'].idxmin()
    
    # Best by combined metric: total_score + normalized validation loss
    # Normalize val_loss to same scale as scores (degrees)
    df_results['combined_score'] = df_results['total_score'] + df_results['best_val_loss']
    best_combined_idx = df_results['combined_score'].idxmin()
    
    # Save best models with descriptive names
    best_models_info = []
    
    for idx, name in [(best_total_idx, 'best_total_score'), 
                      (best_few_idx, 'best_few_shot'),
                      (best_combined_idx, 'best_combined')]:
        result = results[idx]
        config = df_results.iloc[idx]
        
        loss_name = result.get('loss_function', 'L1')
        model_filename = f"gridsearch_{name}_{result['rnn_type']}_{loss_name}_hs{int(config['hidden_size'])}_fc{int(config['fc_units'])}_l{int(config['num_layers'])}.pth"
        
        torch.save({
            'model_state_dict': result['model_state'],
            'config': {
                'lambda_focal': result['lambda_focal'],
                'gamma': result['gamma'],
                'hidden_size': result['hidden_size'],
                'fc_units': result['fc_units'],
                'num_layers': result['num_layers'],
                'use_scheduler': result['use_scheduler'],
                'input_size': input_size,
                'dropout': dropout,
                'rnn_type': result['rnn_type']
            },
            'metrics': {
                'val_loss': result['best_val_loss'],
                'epoch': result['best_epoch'],
                'many': result['many_score'],
                'medium': result['medium_score'],
                'few': result['few_score'],
                'total': config['total_score']
            }
        }, model_filename)
        
        # Add model path to result
        results[idx]['model_path'] = model_filename
        
        best_models_info.append({
            'name': name,
            'file': model_filename,
            'idx': idx,
            'total': config['total_score'],
            'many': config['many'],
            'few': config['few'],
            'val_loss': config['best_val_loss']
        })
        
        print(f"✅ Saved: {model_filename}")
        print(f"   Total: {config['total_score']:.3f}° | Few: {config['few']:.3f}° | Val Loss: {config['best_val_loss']:.5f}")
    
    # Display summary
    print("\n" + "="*80)
    print("📊 GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Total experiments: {len(results)}")
    print(f"Results saved to: grid_search_final.csv")
    print(f"Best models saved: {len(best_models_info)}")
    
    # Sort by different metrics
    print("\n" + "="*80)
    print("🏆 TOP 5 CONFIGURATIONS")
    print("="*80)
    
    print("\n✅ Best by Many-Shot Score:")
    top_many = df_results.nsmallest(5, 'many')
    print(top_many[['rnn_type', 'loss_function', 'lambda_focal', 'gamma', 'hidden_size', 'fc_units', 'num_layers', 
                    'scheduler', 'many', 'medium', 'few']].to_string(index=False))
    
    print("\n✅ Best by Total Score (sum):")
    top_total = df_results.nsmallest(5, 'total_score')
    print(top_total[['rnn_type', 'loss_function', 'lambda_focal', 'gamma', 'hidden_size', 'fc_units', 'num_layers', 
                     'scheduler', 'many', 'medium', 'few', 'total_score']].to_string(index=False))
    
    print("\n✅ Best by Validation Loss:")
    top_val = df_results.nsmallest(5, 'best_val_loss')
    print(top_val[['rnn_type', 'loss_function', 'lambda_focal', 'gamma', 'hidden_size', 'fc_units', 'num_layers', 
                   'scheduler', 'best_val_loss', 'many']].to_string(index=False))
    
    # Visualize results
    visualize_grid_search(df_results, results)
    
    return df_results, results


def visualize_grid_search(df_results, all_results):
    """Create comprehensive visualization of grid search results."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Many-Shot vs Lambda Focal (grouped by scheduler)
    ax1 = plt.subplot(3, 3, 1)
    for sched in df_results['scheduler'].unique():
        subset = df_results[df_results['scheduler'] == sched]
        ax1.scatter(subset['lambda_focal'], subset['many'], 
                   label=f"Scheduler {sched}", alpha=0.6, s=100)
    ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Kaggle Limit')
    ax1.set_xlabel('Lambda Focal', fontsize=10)
    ax1.set_ylabel('Many-Shot MAE (°)', fontsize=10)
    ax1.set_title('Many-Shot vs Lambda Focal', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Total Score vs Hidden Size
    ax2 = plt.subplot(3, 3, 2)
    for h_size in sorted(df_results['hidden_size'].unique()):
        subset = df_results[df_results['hidden_size'] == h_size]
        ax2.scatter(subset.index, subset['total_score'], 
                   label=f"Hidden={h_size}", alpha=0.6, s=100)
    ax2.set_xlabel('Experiment Index', fontsize=10)
    ax2.set_ylabel('Total Score (sum)', fontsize=10)
    ax2.set_title('Total Score by Hidden Size', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Loss vs FC Units
    ax3 = plt.subplot(3, 3, 3)
    for fc in sorted(df_results['fc_units'].unique()):
        subset = df_results[df_results['fc_units'] == fc]
        ax3.scatter(subset.index, subset['best_val_loss'], 
                   label=f"FC={fc}", alpha=0.6, s=100)
    ax3.set_xlabel('Experiment Index', fontsize=10)
    ax3.set_ylabel('Validation Loss', fontsize=10)
    ax3.set_title('Val Loss by FC Units', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Shot-based scores comparison (box plot)
    ax4 = plt.subplot(3, 3, 4)
    shot_data = [df_results['many'], df_results['medium'], df_results['few']]
    bp = ax4.boxplot(shot_data, labels=['Many', 'Medium', 'Few'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax4.set_ylabel('MAE (°)', fontsize=10)
    ax4.set_title('Shot Category Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Scheduler Impact
    ax5 = plt.subplot(3, 3, 5)
    sched_yes = df_results[df_results['scheduler'] == 'YES']
    sched_no = df_results[df_results['scheduler'] == 'NO']
    x = np.arange(3)
    width = 0.35
    if len(sched_yes) > 0:
        means_yes = [sched_yes['many'].mean(), sched_yes['medium'].mean(), sched_yes['few'].mean()]
        ax5.bar(x - width/2, means_yes, width, label='Scheduler ON', color='lightblue')
    if len(sched_no) > 0:
        means_no = [sched_no['many'].mean(), sched_no['medium'].mean(), sched_no['few'].mean()]
        ax5.bar(x + width/2, means_no, width, label='Scheduler OFF', color='lightcoral')
    ax5.set_ylabel('Average MAE (°)', fontsize=10)
    ax5.set_title('Scheduler Impact on Scores', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Many', 'Medium', 'Few'])
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Gamma Impact
    ax6 = plt.subplot(3, 3, 6)
    for gamma in sorted(df_results['gamma'].unique()):
        subset = df_results[df_results['gamma'] == gamma]
        ax6.scatter(subset['lambda_focal'], subset['many'], 
                   label=f"γ={gamma}", alpha=0.6, s=100)
    ax6.axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Lambda Focal', fontsize=10)
    ax6.set_ylabel('Many-Shot MAE (°)', fontsize=10)
    ax6.set_title('Gamma Impact', fontsize=12, fontweight='bold')
    ax6.set_xscale('log')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Num Layers Impact
    ax7 = plt.subplot(3, 3, 7)
    for layers in sorted(df_results['num_layers'].unique()):
        subset = df_results[df_results['num_layers'] == layers]
        ax7.scatter(subset['hidden_size'], subset['total_score'], 
                   label=f"Layers={layers}", alpha=0.6, s=100)
    ax7.set_xlabel('Hidden Size', fontsize=10)
    ax7.set_ylabel('Total Score', fontsize=10)
    ax7.set_title('Num Layers Impact', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Training curve of best model
    ax8 = plt.subplot(3, 3, 8)
    best_idx = df_results['total_score'].idxmin()
    best_result = all_results[best_idx]
    ax8.plot(best_result['train_losses'], label='Train Loss', linewidth=2)
    ax8.plot(best_result['val_losses'], label='Val Loss', linewidth=2)
    ax8.axvline(x=best_result['best_epoch'], color='gray', linestyle='--', alpha=0.7)
    ax8.set_xlabel('Epoch', fontsize=10)
    ax8.set_ylabel('Loss', fontsize=10)
    ax8.set_title('Best Model Training Curve', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # 9. Top 10 configurations table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    top10 = df_results.nsmallest(10, 'total_score')
    table_data = [['Rank', 'λ', 'h', 'fc', 'L', 'Sch', 'Many', 'Total']]
    for rank, (idx, row) in enumerate(top10.iterrows(), 1):
        table_data.append([
            str(rank),
            f"{row['lambda_focal']:.5f}",
            str(int(row['hidden_size'])),
            str(int(row['fc_units'])),
            str(int(row['num_layers'])),
            row['scheduler'],
            f"{row['many']:.3f}",
            f"{row['total_score']:.3f}"
        ])
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.14, 0.08, 0.08, 0.06, 0.08, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    
    for i in range(8):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    table[(1, 6)].set_facecolor('#90EE90')  # Highlight best Many-Shot
    
    ax9.set_title('Top 10 Configurations', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('grid_search_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n💾 Visualization saved to: grid_search_visualization.png")


# ========== USAGE EXAMPLE ==========
"""
from full_grid_search import comprehensive_grid_search, HuberLoss, CombinedLoss

# Define parameter grids
hidden_sizes = [32, 64]
fc_units_list = [32, 64]
num_layers_list = [1, 2, 3]
scheduler_options = [True]
rnn_types = ['RNN', 'GRU']

# Define loss functions to test
loss_configs = [
    {'name': 'L1', 'loss': nn.L1Loss()},
    {'name': 'MSE', 'loss': nn.MSELoss()},
    {'name': 'Huber', 'loss': HuberLoss(delta=1.0)},
    {'name': 'Combined', 'loss': CombinedLoss(alpha_mse=0.4, alpha_l1=0.3, alpha_huber=0.3)},
]

# Run comprehensive grid search
df_results, all_results = comprehensive_grid_search(
    # Required parameters
    hidden_sizes=hidden_sizes,
    fc_units_list=fc_units_list,
    num_layers_list=num_layers_list,
    use_scheduler_options=scheduler_options,
    
    # Optional: RNN types and loss functions
    rnn_types=rnn_types,
    loss_function_configs=loss_configs,
    
    # Training configuration
    num_epochs=50,
    early_stop_thresh=10,
    train_loader=train_loader,
    val_loader=val_loader,
    model_class=RNN,
    criterion=nn.L1Loss(),  # Will be overridden by loss_configs
    lr=0.0003,
    weight_decay=1e-4,
    scheduler_params={
        'mode': 'min',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-6
    },
    device=device,
    input_size=input_size,
    dropout=[0, 0, 0],
    
    # Optional controls
    max_experiments=None,  # Run all experiments (or limit to N random samples)
    save_every_n=5  # Save intermediate results every 5 experiments
)

# Total combinations: 2 × 2 × 3 × 1 × 2 × 4 = 96 experiments
# Tests: 2 hidden sizes × 2 fc_units × 3 layers × 1 scheduler × 2 RNN types × 4 loss functions
"""
