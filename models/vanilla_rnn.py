"""Vanilla RNN - Winning architecture for sideslip estimation."""

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """Simple RNN with ReLU. Best: 64 hidden, 1 layer, no dropout."""
    
    def __init__(self, input_size=22, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          nonlinearity='relu', dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(self.dropout(h[-1]))
