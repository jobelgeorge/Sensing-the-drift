"""
Visualization Utilities for Sideslip Estimation
================================================

Plotly-based interactive visualizations for:
- Training curves (loss, metrics)
- Hyperparameter sweep results
- Model comparison dashboards
- Prediction analysis

Based on the project's established visualization patterns.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# 1. TRAINING CURVES
# ============================================================================
def plot_training_curves(history: Dict[str, List[float]], 
                         title: str = "Training Progress",
                         stage_boundary: Optional[int] = None) -> go.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss' keys (and optionally stage-specific)
        title: Plot title
        stage_boundary: Epoch number where Stage 2 begins (for two-stage training)
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> history = {'train_loss': [...], 'val_loss': [...]}
        >>> fig = plot_training_curves(history, stage_boundary=30)
        >>> fig.show()
    """
    fig = go.Figure()
    
    # Handle two-stage format
    if 'stage1_train' in history:
        train_loss = history['stage1_train'] + history.get('stage2_train', [])
        val_loss = history['stage1_val'] + history.get('stage2_val', [])
        if stage_boundary is None:
            stage_boundary = len(history['stage1_train'])
    else:
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
    
    epochs = list(range(1, len(train_loss) + 1))
    
    # Training loss
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        mode='lines',
        name='Train Loss',
        line=dict(color='#636EFA', width=2)
    ))
    
    # Validation loss
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        mode='lines',
        name='Val Loss',
        line=dict(color='#EF553B', width=2)
    ))
    
    # Stage boundary
    if stage_boundary is not None:
        fig.add_vline(
            x=stage_boundary, 
            line_dash="dash", 
            line_color="green",
            annotation_text="Stage 2",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_two_stage_comparison(histories: Dict[str, Dict], 
                              labels: Optional[List[str]] = None) -> go.Figure:
    """
    Compare training curves from multiple two-stage training runs.
    
    Args:
        histories: Dict mapping run names to history dicts
        labels: Optional custom labels for runs
    
    Returns:
        Plotly Figure with subplots
    """
    n_runs = len(histories)
    fig = make_subplots(
        rows=1, cols=n_runs,
        subplot_titles=list(histories.keys()) if labels is None else labels,
        shared_yaxes=True
    )
    
    colors = px.colors.qualitative.Plotly
    
    for i, (name, history) in enumerate(histories.items(), 1):
        if 'stage1_train' in history:
            train = history['stage1_train'] + history.get('stage2_train', [])
            val = history['stage1_val'] + history.get('stage2_val', [])
            boundary = len(history['stage1_train'])
        else:
            train = history.get('train_loss', [])
            val = history.get('val_loss', [])
            boundary = None
        
        fig.add_trace(
            go.Scatter(y=train, name='Train', line=dict(color=colors[0]), showlegend=(i==1)),
            row=1, col=i
        )
        fig.add_trace(
            go.Scatter(y=val, name='Val', line=dict(color=colors[1]), showlegend=(i==1)),
            row=1, col=i
        )
        
        if boundary:
            fig.add_vline(x=boundary, line_dash="dash", line_color="green", row=1, col=i)
    
    fig.update_layout(
        title='Two-Stage Training Comparison',
        height=400,
        template='plotly_white'
    )
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text='Loss', col=1)
    
    return fig


# ============================================================================
# 2. HYPERPARAMETER SWEEP VISUALIZATION
# ============================================================================
def plot_sweep_results(sweep_df: pd.DataFrame,
                       x_col: str,
                       y_col: str = 'overall_mae',
                       title: str = "Hyperparameter Sweep",
                       highlight_optimal: bool = True) -> go.Figure:
    """
    Plot hyperparameter sweep results with optimal point highlighted.
    
    Args:
        sweep_df: DataFrame with sweep results
        x_col: Column name for x-axis (hyperparameter)
        y_col: Column name for y-axis (metric)
        title: Plot title
        highlight_optimal: Whether to highlight the optimal point
    
    Returns:
        Plotly Figure
    
    Example:
        >>> df = pd.DataFrame({'threshold': [1.5, 2.0, 2.5], 'overall_mae': [0.23, 0.21, 0.24]})
        >>> fig = plot_sweep_results(df, 'threshold')
        >>> fig.show()
    """
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=sweep_df[x_col],
        y=sweep_df[y_col],
        mode='lines+markers',
        name=y_col,
        line=dict(color='#636EFA', width=3),
        marker=dict(size=10)
    ))
    
    # Highlight optimal
    if highlight_optimal:
        opt_idx = sweep_df[y_col].idxmin()
        opt_row = sweep_df.loc[opt_idx]
        
        fig.add_trace(go.Scatter(
            x=[opt_row[x_col]],
            y=[opt_row[y_col]],
            mode='markers',
            name=f'Optimal ({opt_row[y_col]:.4f})',
            marker=dict(size=18, color='red', symbol='star')
        ))
        
        fig.add_annotation(
            x=opt_row[x_col],
            y=opt_row[y_col],
            text=f"Best: {opt_row[y_col]:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-40
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_sweep_dashboard(sweep_results: Dict[str, pd.DataFrame],
                         metric_col: str = 'overall_mae') -> go.Figure:
    """
    Create a dashboard showing multiple hyperparameter sweeps.
    
    Args:
        sweep_results: Dict mapping sweep names to DataFrames
        metric_col: Column name for the metric to plot
    
    Returns:
        Plotly Figure with subplots
    
    Example:
        >>> sweeps = {
        ...     'threshold': df_threshold,
        ...     'slope': df_slope,
        ...     'epochs': df_epochs
        ... }
        >>> fig = plot_sweep_dashboard(sweeps)
        >>> fig.show()
    """
    n_sweeps = len(sweep_results)
    n_cols = min(3, n_sweeps)
    n_rows = (n_sweeps + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=list(sweep_results.keys()),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    colors = px.colors.qualitative.Plotly
    
    for i, (name, df) in enumerate(sweep_results.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Get x column (first non-metric column)
        x_col = [c for c in df.columns if c != metric_col][0]
        
        # Main trace
        fig.add_trace(
            go.Scatter(
                x=df[x_col], y=df[metric_col],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
                name=name,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Optimal marker
        opt_idx = df[metric_col].idxmin()
        fig.add_trace(
            go.Scatter(
                x=[df.loc[opt_idx, x_col]],
                y=[df.loc[opt_idx, metric_col]],
                mode='markers',
                marker=dict(size=14, color='red', symbol='star'),
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text=x_col, row=row, col=col)
        fig.update_yaxes(title_text='MAE', row=row, col=col)
    
    fig.update_layout(
        title='<b>Hyperparameter Sweep Dashboard</b>',
        height=350 * n_rows,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# 3. MODEL COMPARISON
# ============================================================================
def plot_method_comparison(results: List[Dict[str, Any]],
                           title: str = "Method Comparison") -> go.Figure:
    """
    Bar chart comparing different methods/configurations.
    
    Args:
        results: List of dicts with 'method', 'score', and optionally 'color'
        title: Plot title
    
    Returns:
        Plotly Figure
    
    Example:
        >>> results = [
        ...     {'method': 'L1 Only', 'score': 0.80},
        ...     {'method': 'Two-Stage', 'score': 0.21, 'highlight': True}
        ... ]
        >>> fig = plot_method_comparison(results)
        >>> fig.show()
    """
    df = pd.DataFrame(results)
    
    colors = ['#636EFA'] * len(df)
    if 'highlight' in df.columns:
        for i, row in df.iterrows():
            if row.get('highlight', False):
                colors[i] = '#00CC96'
    
    # Find best
    best_idx = df['score'].idxmin()
    colors[best_idx] = '#00CC96'
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['method'],
            y=df['score'],
            marker_color=colors,
            text=[f"{s:.4f}" for s in df['score']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Method',
        yaxis_title='Score (MAE)',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_results_table(results_df: pd.DataFrame,
                       highlight_best: bool = True,
                       metric_col: str = 'overall_mae') -> go.Figure:
    """
    Create an interactive table visualization of results.
    
    Args:
        results_df: DataFrame with results
        highlight_best: Whether to highlight the best row
        metric_col: Column to use for determining best
    
    Returns:
        Plotly Figure with table
    """
    # Prepare colors
    fill_colors = [['white'] * len(results_df) for _ in range(len(results_df.columns))]
    
    if highlight_best and metric_col in results_df.columns:
        best_idx = results_df[metric_col].idxmin()
        for j in range(len(results_df.columns)):
            fill_colors[j][best_idx] = '#90EE90'  # Light green
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(results_df.columns),
            fill_color='#636EFA',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[results_df[col] for col in results_df.columns],
            fill_color=fill_colors,
            align='center',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title='Results Summary',
        height=max(300, 50 * len(results_df) + 100)
    )
    
    return fig


# ============================================================================
# 4. PREDICTION ANALYSIS
# ============================================================================
def plot_predictions_scatter(predictions: np.ndarray,
                             targets: np.ndarray,
                             band_boundaries: Tuple[float, float] = (2.4, 4.0),
                             title: str = "Predictions vs Targets") -> go.Figure:
    """
    Scatter plot of predictions vs targets with band regions.
    
    Args:
        predictions: Model predictions
        targets: True values
        band_boundaries: (many/medium boundary, medium/few boundary)
        title: Plot title
    
    Returns:
        Plotly Figure
    """
    # Create band labels
    bands = np.where(
        np.abs(targets) <= band_boundaries[0], 'many-shot',
        np.where(np.abs(targets) <= band_boundaries[1], 'medium-shot', 'few-shot')
    )
    
    df = pd.DataFrame({
        'Target': targets,
        'Prediction': predictions,
        'Band': bands,
        'Error': np.abs(predictions - targets)
    })
    
    fig = px.scatter(
        df, x='Target', y='Prediction',
        color='Band',
        color_discrete_map={
            'many-shot': '#636EFA',
            'medium-shot': '#EF553B', 
            'few-shot': '#00CC96'
        },
        hover_data=['Error'],
        title=title
    )
    
    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        template='plotly_white',
        height=600
    )
    
    return fig


def plot_error_distribution(predictions: np.ndarray,
                            targets: np.ndarray,
                            title: str = "Error Distribution") -> go.Figure:
    """
    Histogram of prediction errors.
    
    Args:
        predictions: Model predictions
        targets: True values
        title: Plot title
    
    Returns:
        Plotly Figure
    """
    errors = predictions - targets
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Errors',
        marker_color='#636EFA'
    ))
    
    # Add mean line
    mean_error = np.mean(errors)
    fig.add_vline(
        x=mean_error,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_error:.4f}"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Prediction Error',
        yaxis_title='Count',
        template='plotly_white',
        height=400
    )
    
    return fig


# ============================================================================
# 5. HEATMAP VISUALIZATION
# ============================================================================
def plot_grid_search_heatmap(grid_df: pd.DataFrame,
                             x_col: str,
                             y_col: str,
                             value_col: str = 'overall_mae',
                             title: str = "Grid Search Results") -> go.Figure:
    """
    Create a heatmap for 2D grid search results.
    
    Args:
        grid_df: DataFrame with grid search results
        x_col: Column for x-axis
        y_col: Column for y-axis
        value_col: Column for values (color)
        title: Plot title
    
    Returns:
        Plotly Figure
    
    Example:
        >>> df = pd.DataFrame({
        ...     's1_epochs': [25, 25, 30, 30],
        ...     's2_epochs': [35, 40, 35, 40],
        ...     'overall_mae': [0.23, 0.22, 0.22, 0.21]
        ... })
        >>> fig = plot_grid_search_heatmap(df, 's2_epochs', 's1_epochs')
        >>> fig.show()
    """
    pivot = grid_df.pivot(index=y_col, columns=x_col, values=value_col)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale='RdYlGn_r',
        text=np.round(pivot.values, 4),
        texttemplate='%{text}',
        textfont=dict(size=12),
        hovertemplate=f'{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>MAE=%{{z:.4f}}<extra></extra>'
    ))
    
    # Find and annotate optimal
    min_val = pivot.values.min()
    min_idx = np.unravel_index(pivot.values.argmin(), pivot.values.shape)
    
    fig.add_annotation(
        x=min_idx[1],
        y=min_idx[0],
        text="★ BEST",
        showarrow=True,
        arrowhead=2,
        arrowcolor="black",
        font=dict(size=12, color="black"),
        ax=40,
        ay=-30
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=500
    )
    
    return fig


# ============================================================================
# 6. PREDICTIONS VS TRUTH LINE PLOT
# ============================================================================
def plot_predictions_vs_truth(predictions: np.ndarray,
                              targets: np.ndarray,
                              title: str = "Ground Truth vs Predictions",
                              max_samples: Optional[int] = None,
                              height: int = 500,
                              width: int = 1200) -> go.Figure:
    """
    Interactive line plot showing predictions overlaid on ground truth.
    
    Creates a Plotly figure with ground truth in blue and predictions in red,
    allowing zooming, panning, and hovering to examine specific values.
    
    Args:
        predictions: Model predictions array
        targets: Ground truth values array  
        title: Plot title
        max_samples: Maximum samples to plot (None = all). Use for large datasets.
        height: Plot height in pixels
        width: Plot width in pixels
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> preds, targets = get_predictions(model, val_loader, device)
        >>> fig = plot_predictions_vs_truth(preds, targets, title="Model Predictions")
        >>> fig.show()
    """
    # Flatten arrays
    preds_flat = np.array(predictions).flatten()
    targets_flat = np.array(targets).flatten()
    
    # Subsample if needed
    if max_samples is not None and len(preds_flat) > max_samples:
        indices = np.linspace(0, len(preds_flat)-1, max_samples, dtype=int)
        preds_flat = preds_flat[indices]
        targets_flat = targets_flat[indices]
    
    # Calculate statistics
    mae = np.mean(np.abs(preds_flat - targets_flat))
    
    fig = go.Figure()
    
    # Ground truth
    fig.add_trace(go.Scatter(
        y=targets_flat,
        mode='lines',
        name='Ground Truth',
        line=dict(color='blue', width=1.5),
        hovertemplate='Index: %{x}<br>Truth: %{y:.3f}<extra></extra>'
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        y=preds_flat,
        mode='lines',
        name='Predictions',
        line=dict(color='red', width=1.5),
        opacity=0.8,
        hovertemplate='Index: %{x}<br>Pred: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{title} (MAE: {mae:.4f})',
        xaxis_title='Sample Index',
        yaxis_title='Sideslip Angle (degrees)',
        width=width,
        height=height,
        template='plotly_white',
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    return fig


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
def demo_visualizations():
    """
    Demonstrate all visualization functions with sample data.
    """
    print("="*60)
    print("VISUALIZATION UTILITIES DEMO")
    print("="*60)
    
    # 1. Training curves
    history = {
        'stage1_train': list(np.linspace(0.5, 0.2, 30)),
        'stage1_val': list(np.linspace(0.45, 0.18, 30)),
        'stage2_train': list(np.linspace(0.2, 0.12, 40)),
        'stage2_val': list(np.linspace(0.18, 0.11, 40))
    }
    fig1 = plot_training_curves(history, "Two-Stage Training Progress", stage_boundary=30)
    print("1. Training curves: ✓")
    
    # 2. Sweep results
    sweep_df = pd.DataFrame({
        'threshold': [1.5, 1.8, 2.0, 2.2, 2.5],
        'overall_mae': [0.232, 0.225, 0.217, 0.221, 0.228]
    })
    fig2 = plot_sweep_results(sweep_df, 'threshold', title='Threshold Sweep')
    print("2. Sweep results: ✓")
    
    # 3. Method comparison
    results = [
        {'method': 'L1 Only', 'score': 0.80},
        {'method': 'WeightedL1 Only', 'score': 0.25},
        {'method': 'Two-Stage', 'score': 0.217}
    ]
    fig3 = plot_method_comparison(results)
    print("3. Method comparison: ✓")
    
    # 4. Predictions scatter
    np.random.seed(42)
    targets = np.random.randn(500) * 3
    predictions = targets + np.random.randn(500) * 0.3
    fig4 = plot_predictions_scatter(predictions, targets)
    print("4. Predictions scatter: ✓")
    
    # 5. Grid search heatmap
    grid_df = pd.DataFrame({
        's1_epochs': [25, 25, 25, 30, 30, 30, 35, 35, 35],
        's2_epochs': [35, 40, 45, 35, 40, 45, 35, 40, 45],
        'overall_mae': [0.23, 0.22, 0.24, 0.22, 0.217, 0.23, 0.225, 0.22, 0.235]
    })
    fig5 = plot_grid_search_heatmap(grid_df, 's2_epochs', 's1_epochs')
    print("5. Grid search heatmap: ✓")
    
    print("="*60)
    print("All visualizations ready!")
    print("Usage: from utils.visualization import *")
    print("="*60)
    
    return fig1, fig2, fig3, fig4, fig5


if __name__ == "__main__":
    demo_visualizations()
