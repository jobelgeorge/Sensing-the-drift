"""Simple GRU for architecture comparison."""

import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    """GRU baseline for comparison."""
    
    def __init__(self, input_size=22, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(self.dropout(h[-1]))
