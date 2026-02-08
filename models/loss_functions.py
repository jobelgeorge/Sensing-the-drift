"""
Loss Functions for Deep Imbalanced Regression
==============================================

Collection of loss functions tested for vehicle sideslip angle prediction
under data imbalance. Includes standard losses, weighted variants, 
focal mechanisms, and combined objectives.

References:
- Focal Loss: Lin et al. (2017) "Focal Loss for Dense Object Detection"
- Deep Imbalanced Regression: Yang & Zha (2021) "Delving into Deep Imbalanced Regression"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FOCAL LOSS (Best Performer)
# =============================================================================

class FocalL1Loss(nn.Module):
    """
    Focal L1 Loss - Adaptive error-based sample weighting.
    
    Applies focal mechanism to L1 loss for continuous error-based focusing.
    Best performing loss for imbalanced regression.
    
    Args:
        gamma (float): Focusing parameter. Higher values focus more on hard samples.
                      Optimal: 0.5 for this dataset.
        lambda_reg (float): L2 regularization coefficient. Default: 0.0 (not needed).
        
    Loss Formula:
        L = (1/N) * sum(|y - ŷ|^(1+γ)) + λ * ||ŷ||²
        
    Best Configuration:
        gamma=0.5, lambda_reg=0.0
        Achieves 0.1963 MAE (20.2% improvement over L1)
    """
    
    def __init__(self, gamma=0.5, lambda_reg=0.0):
        super().__init__()
        self.gamma = gamma
        self.lambda_reg = lambda_reg
    
    def forward(self, pred, target, shot_bands=None):
        """
        Args:
            pred: Predictions (batch_size, 1)
            target: Ground truth (batch_size, 1)
            shot_bands: Optional shot-band indices (not used in focal, for API compatibility)
        """
        abs_error = torch.abs(pred - target)
        focal_weight = (abs_error + 1e-6) ** self.gamma
        focal_loss = abs_error * focal_weight
        
        reg_loss = self.lambda_reg * (pred ** 2).mean() if self.lambda_reg > 0 else 0
        return focal_loss.mean() + reg_loss


class AdaptiveFocalL1Loss(nn.Module):
    """
    Focal L1 with adaptive gamma adjustment during training.
    
    Dynamically adjusts gamma based on few-shot performance to maintain
    balance between shot-bands.
    
    Args:
        gamma_start (float): Initial gamma value
        gamma_end (float): Final gamma value
        adaptive_gamma (bool): Enable gamma updates during training
        shot_weights (tensor): Optional per-band weights
        lambda_reg (float): Regularization coefficient
    """
    
    def __init__(self, gamma_start=1.0, gamma_end=2.0, adaptive_gamma=True,
                 shot_weights=None, lambda_reg=0.0):
        super().__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.gamma = gamma_start
        self.adaptive_gamma = adaptive_gamma
        self.shot_weights = shot_weights or torch.tensor([1.2, 1.1, 1.0])
        self.lambda_reg = lambda_reg
    
    def update_gamma(self, epoch, max_epochs):
        """Update gamma based on training progress"""
        if self.adaptive_gamma:
            progress = float(epoch) / float(max_epochs)
            self.gamma = self.gamma_start + progress * (self.gamma_end - self.gamma_start)
    
    def update_gamma_band(self, few_shot_error, target_few_shot_error=0.5, gamma_step=0.05):
        """Update gamma based on few-shot performance"""
        if few_shot_error > target_few_shot_error:
            self.gamma = min(self.gamma + gamma_step, self.gamma_end)
        else:
            self.gamma = max(self.gamma - gamma_step, self.gamma_start)
    
    def forward(self, pred, target, shot_bands=None):
        abs_error = torch.abs(pred - target)
        
        # Focal weighting
        focal_weight = (abs_error + 1e-6) ** self.gamma
        focal_loss = abs_error * focal_weight
        
        # Optional shot-band weighting
        if shot_bands is not None:
            weights = self.shot_weights.to(pred.device)[shot_bands]
            focal_loss = focal_loss.squeeze() * weights
        
        reg_loss = self.lambda_reg * (pred ** 2).mean() if self.lambda_reg > 0 else 0
        return focal_loss.mean() + reg_loss


# =============================================================================
# WEIGHTED LOSSES
# =============================================================================

class WeightedL1Loss(nn.Module):
    """
    L1 Loss with shot-band specific weights.
    
    Applies discrete weights based on shot-band membership to balance
    performance across many/medium/few-shot samples.
    
    Args:
        weights (tensor): Weights for [many, medium, few] shot-bands.
                         Default: [1.2, 1.1, 1.0] (based on error analysis)
        scheme (str): Weighting scheme name (for logging)
        
    Performance:
        ~11.2% improvement over baseline L1
        Worse than Focal L1 (continuous vs discrete weighting)
    """
    
    def __init__(self, weights=None, scheme='fixed'):
        super().__init__()
        if weights is None:
            weights = torch.tensor([1.2, 1.1, 1.0])
        self.register_buffer('weights', weights)
        self.scheme = scheme
    
    def forward(self, pred, target, shot_bands):
        abs_error = torch.abs(pred - target)
        weights = self.weights.to(pred.device)[shot_bands].unsqueeze(1)
        weighted_error = abs_error * weights
        return weighted_error.mean()


class AdaptiveWeightedL1Loss(nn.Module):
    """
    Weighted L1 with dynamic weight adjustment.
    
    Updates weights during training based on current shot-band performance
    relative to target MAE values.
    
    Args:
        initial_weights (tensor): Starting weights for each shot-band
        adaptation_rate (float): Learning rate for weight updates
        target_maes (dict): Target MAE for each shot-band
    """
    
    def __init__(self, initial_weights=None, adaptation_rate=0.1,
                 target_maes=None):
        super().__init__()
        if initial_weights is None:
            initial_weights = torch.tensor([1.0, 1.0, 1.0])
        self.weights = nn.Parameter(initial_weights, requires_grad=False)
        self.adaptation_rate = adaptation_rate
        
        if target_maes is None:
            target_maes = {'many': 0.15, 'medium': 0.20, 'few': 0.35}
        self.target_maes = torch.tensor([target_maes['many'], 
                                        target_maes['medium'], 
                                        target_maes['few']])
    
    def forward(self, pred, target, shot_bands):
        abs_error = torch.abs(pred - target)
        weights = self.weights.to(pred.device)[shot_bands].unsqueeze(1)
        weighted_error = abs_error * weights
        return weighted_error.mean()
    
    def update_weights(self, band_maes):
        """Update weights based on current performance"""
        current_maes = torch.tensor([band_maes['many'], 
                                     band_maes['medium'], 
                                     band_maes['few']])
        target_maes = self.target_maes.to(current_maes.device)
        
        # Calculate relative error
        rel_error = (current_maes - target_maes) / target_maes
        
        # Adjust weights
        adjustment = 1.0 + self.adaptation_rate * rel_error
        self.weights.data *= adjustment
        
        # Normalize
        self.weights.data /= self.weights.data.min()


# =============================================================================
# COMBINED LOSSES
# =============================================================================

class MSE_L1_Loss(nn.Module):
    """
    Combination of MSE and L1 (Elastic Loss).
    
    Args:
        alpha (float): Weight for MSE. L1 weight = 1 - alpha
                      Optimal: 0.3 (favor L1 for robustness)
        
    Performance:
        ~12.5% improvement over L1
        Still worse than Focal L1
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = 1.0 - alpha
    
    def forward(self, pred, target, shot_bands=None):
        mse = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        return self.alpha * mse + self.beta * l1


class Huber_L1_Loss(nn.Module):
    """
    Combination of Huber and L1 losses.
    
    Args:
        alpha (float): Weight for Huber
        delta (float): Huber threshold parameter
    """
    
    def __init__(self, alpha=0.5, delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.delta = delta
    
    def forward(self, pred, target, shot_bands=None):
        huber = F.huber_loss(pred, target, delta=self.delta)
        l1 = F.l1_loss(pred, target)
        return self.alpha * huber + self.beta * l1


class TripleLoss(nn.Module):
    """
    Three-way combination: MSE + L1 + Huber.
    
    Args:
        alpha, beta, gamma: Weights for MSE, L1, Huber (auto-normalized)
        delta: Huber threshold
        
    Note:
        No significant advantage over simpler MSE+L1.
        Added complexity not justified by performance.
    """
    
    def __init__(self, alpha=0.33, beta=0.34, gamma=0.33, delta=1.0):
        super().__init__()
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.delta = delta
    
    def forward(self, pred, target, shot_bands=None):
        mse = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        huber = F.huber_loss(pred, target, delta=self.delta)
        return self.alpha * mse + self.beta * l1 + self.gamma * huber


class HybridWeightedFocalL1Loss(nn.Module):
    """
    Combination of shot-band weighting and focal mechanism.
    
    Args:
        weights (tensor): Shot-band weights
        gamma (float): Focal gamma parameter
        
    WARNING:
        Performance degrades compared to Focal L1 alone.
        Over-correction from combining discrete and continuous balancing.
        NOT recommended - use pure Focal L1 instead.
    """
    
    def __init__(self, weights=None, gamma=0.5):
        super().__init__()
        if weights is None:
            weights = torch.tensor([1.2, 1.1, 1.0])
        self.register_buffer('weights', weights)
        self.gamma = gamma
    
    def forward(self, pred, target, shot_bands):
        abs_error = torch.abs(pred - target)
        
        # Shot-band weighting
        shot_weights = self.weights.to(pred.device)[shot_bands].unsqueeze(1)
        
        # Focal weighting
        focal_weight = (abs_error + 1e-6) ** self.gamma
        
        # Combined (causes over-correction)
        combined_loss = abs_error * shot_weights * focal_weight
        return combined_loss.mean()


class Focal_MSE_Loss(nn.Module):
    """
    Combination of Focal L1 and MSE.
    
    Args:
        alpha (float): Weight for Focal L1
        focal_gamma (float): Focal gamma parameter
        
    WARNING:
        MSE component degrades Focal L1 performance.
        Even small MSE weights (0.2) reduce effectiveness.
        NOT recommended - use pure Focal L1.
    """
    
    def __init__(self, alpha=0.7, focal_gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, pred, target, shot_bands=None):
        # Focal L1
        abs_error = torch.abs(pred - target)
        focal_weight = (abs_error + 1e-6) ** self.focal_gamma
        focal_l1 = (abs_error * focal_weight).mean()
        
        # MSE
        mse = F.mse_loss(pred, target)
        
        return self.alpha * focal_l1 + self.beta * mse


# =============================================================================
# STANDARD LOSSES (for comparison)
# =============================================================================

class HuberLoss(nn.Module):
    """
    Huber Loss - L2 for small errors, L1 for large.
    
    Args:
        delta (float): Threshold for switching between L2 and L1.
                      Optimal: 1.0
                      
    Performance:
        ~4.3% improvement over L1
        Not needed for this dataset (few-shot already performs well)
    """
    
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target, shot_bands=None):
        return F.huber_loss(pred, target, delta=self.delta)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_shot_band_weights(train_df, method='inverse_freq'):
    """
    Calculate shot-band weights from training data.
    
    Args:
        train_df (DataFrame): Training data with 'shot_band' column
        method (str): Weighting method
            - 'fixed': Empirical weights [1.2, 1.1, 1.0]
            - 'inverse_freq': Inverse frequency weighting
            - 'class_balanced': Effective number (Cui et al. 2019)
            - 'sqrt_inv_freq': Square root of inverse frequency
            
    Returns:
        torch.Tensor: Weights for [many, medium, few] shot-bands
    """
    shot_counts = train_df['shot_band'].value_counts()
    total = len(train_df)
    
    if method == 'fixed':
        return torch.tensor([1.2, 1.1, 1.0])
    
    elif method == 'inverse_freq':
        inv_freq = torch.tensor([
            total / shot_counts['many'],
            total / shot_counts['medium'],
            total / shot_counts['few']
        ])
        return inv_freq / inv_freq.min()  # Normalize
    
    elif method == 'class_balanced':
        beta = 0.999
        eff_num = torch.tensor([
            (1 - beta) / (1 - beta ** shot_counts['many']),
            (1 - beta) / (1 - beta ** shot_counts['medium']),
            (1 - beta) / (1 - beta ** shot_counts['few'])
        ])
        return eff_num / eff_num.min()
    
    elif method == 'sqrt_inv_freq':
        inv_freq = torch.tensor([
            total / shot_counts['many'],
            total / shot_counts['medium'],
            total / shot_counts['few']
        ])
        sqrt_inv = torch.sqrt(inv_freq)
        return sqrt_inv / sqrt_inv.min()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def get_loss_function(loss_type='focal_l1', **kwargs):
    """
    Factory function for creating loss functions.
    
    Args:
        loss_type (str): Type of loss function
            - 'l1': Standard L1
            - 'mse': Mean Squared Error
            - 'huber': Huber Loss
            - 'focal_l1': Focal L1 (RECOMMENDED)
            - 'weighted_l1': Weighted L1
            - 'adaptive_weighted_l1': Adaptive Weighted L1
            - 'mse_l1': MSE + L1 combination
            - 'huber_l1': Huber + L1 combination
            - 'triple': MSE + L1 + Huber
        **kwargs: Arguments passed to loss constructor
        
    Returns:
        Loss function instance
        
    Example:
        >>> loss_fn = get_loss_function('focal_l1', gamma=0.5)
        >>> loss = loss_fn(predictions, targets)
    """
    loss_map = {
        'l1': nn.L1Loss,
        'mse': nn.MSELoss,
        'huber': HuberLoss,
        'focal_l1': FocalL1Loss,
        'weighted_l1': WeightedL1Loss,
        'adaptive_weighted_l1': AdaptiveWeightedL1Loss,
        'mse_l1': MSE_L1_Loss,
        'huber_l1': Huber_L1_Loss,
        'triple': TripleLoss,
        'hybrid_focal': HybridWeightedFocalL1Loss,
        'focal_mse': Focal_MSE_Loss,
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_type](**kwargs)


# =============================================================================
# RECOMMENDED CONFIGURATION
# =============================================================================

"""
RECOMMENDED LOSS FUNCTION FOR IMBALANCED REGRESSION:

loss_fn = FocalL1Loss(gamma=0.5, lambda_reg=0.0)

Why Focal L1 with gamma=0.5:
1. 20.2% improvement over baseline L1 (0.2464 → 0.1963 MAE)
2. Balanced improvements across all shot-bands:
   - Many-shot: 27% improvement (0.193 → 0.141)
   - Medium-shot: 21% improvement (0.256 → 0.201)
   - Few-shot: 16% improvement (0.339 → 0.285)
3. Single hyperparameter (gamma) - simple to tune
4. Continuous error-based adaptation > discrete shot-band weights
5. No additional regularization needed

Alternative (if Focal unavailable):
loss_fn = WeightedL1Loss(weights=calculate_shot_band_weights(train_df, 'inverse_freq'))
Performance: ~11% improvement over L1
"""
