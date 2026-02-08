"""
Models Module - Sideslip Estimation Challenge
=============================================

Neural network architectures for vehicle sideslip estimation.
"""

from models.vanilla_rnn import VanillaRNN
from models.residual_rnn import InputResidualRNN, HiddenResidualRNN, DeepResidualRNN
from models.simple_lstm import SimpleLSTM
from models.bilstm_attention import BiLSTMAttention
from models.simple_gru import SimpleGRU
from models.simple_bilstm import SimpleBiLSTM
from models.vae import SideslipVAE
from models.uvote import UVOTELSTM

# Loss functions
from models.loss_functions import (
    FocalL1Loss,
    AdaptiveFocalL1Loss,
    WeightedL1Loss,
    AdaptiveWeightedL1Loss,
    MSE_L1_Loss,
    Huber_L1_Loss,
    TripleLoss,
    HybridWeightedFocalL1Loss,
    Focal_MSE_Loss,
    HuberLoss,
    get_loss_function,
    calculate_shot_band_weights,
)

__all__ = [
    # Models
    'VanillaRNN',      # Winning architecture (~13K params)
    'InputResidualRNN',
    'HiddenResidualRNN',
    'DeepResidualRNN',
    'SimpleLSTM',
    'BiLSTMAttention',
    'SimpleGRU',
    'SimpleBiLSTM',
    'SideslipVAE',
    'UVOTELSTM',
    # Loss functions
    'FocalL1Loss',
    'AdaptiveFocalL1Loss',
    'WeightedL1Loss',
    'AdaptiveWeightedL1Loss',
    'MSE_L1_Loss',
    'Huber_L1_Loss',
    'TripleLoss',
    'HybridWeightedFocalL1Loss',
    'Focal_MSE_Loss',
    'HuberLoss',
    'get_loss_function',
    'calculate_shot_band_weights',
]
