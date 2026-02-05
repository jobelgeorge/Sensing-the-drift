"""
Models Module - Sideslip Estimation Challenge
=============================================

Neural network architectures for vehicle sideslip estimation.
"""

from models.vanilla_rnn import VanillaRNN
from models.simple_lstm import SimpleLSTM
from models.bilstm_attention import BiLSTMAttention
from models.simple_gru import SimpleGRU

__all__ = [
    'VanillaRNN',      # Winning architecture (~13K params)
    'SimpleLSTM',
    'BiLSTMAttention',
    'SimpleGRU',
]
