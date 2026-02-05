"""
Utils Module - Sideslip Estimation Challenge
============================================

Core utilities for training, visualization, and data loading.
"""

from utils.training import train_two_stage, train_single_stage, evaluate, get_predictions
from utils.visualization import (
    plot_training_curves, plot_sweep_results, plot_sweep_dashboard,
    plot_method_comparison, plot_predictions_scatter, plot_error_distribution,
    plot_grid_search_heatmap
)
from utils.weighted_l1_loss import WeightedL1Loss
from utils.reproducibility_utils import seed_everything
from utils.data_loading import load_data, load_test_data
from utils.band_mapping import load_band_mapping, calculate_band_mae

__all__ = [
    # Training
    'train_two_stage', 'train_single_stage', 'evaluate', 'get_predictions',
    # Visualization
    'plot_training_curves', 'plot_sweep_results', 'plot_sweep_dashboard',
    'plot_method_comparison', 'plot_predictions_scatter', 'plot_error_distribution',
    'plot_grid_search_heatmap',
    # Loss
    'WeightedL1Loss',
    # Utilities
    'seed_everything', 'load_data', 'load_test_data',
    'load_band_mapping', 'calculate_band_mae',
]
