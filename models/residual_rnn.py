"""Residual RNN variants for sideslip estimation.

Implements two types of residual connections:
1. InputResidualRNN: Skip connection from input to output
2. HiddenResidualRNN: Residual connections in RNN hidden states
"""

import torch
import torch.nn as nn


class InputResidualRNN(nn.Module):
    """VanillaRNN with input-to-output skip connection.
    
    Architecture: output = FC(RNN(x)) + Projection(x_mean)
    The mean-pooled input is projected and added to the RNN output.
    """
    
    def __init__(self, input_size=22, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Main RNN branch
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          nonlinearity='relu', dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Residual projection: input -> 1D (skip connection)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 1)
        )
    
    def forward(self, x):
        # Main RNN path
        _, h = self.rnn(x)
        rnn_out = self.fc(self.dropout(h[-1]))
        
        # Residual path: mean-pool over time, then project
        x_mean = x.mean(dim=1)  # [batch, seq, features] -> [batch, features]
        residual = self.input_proj(x_mean)
        
        return rnn_out + residual


class HiddenResidualRNN(nn.Module):
    """VanillaRNN with residual connections in hidden states.
    
    For multi-layer RNNs, adds h_prev to h_current at each layer.
    For single-layer, adds identity mapping via projection.
    """
    
    def __init__(self, input_size=22, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if num_layers == 1:
            # For single layer, use input projection as residual
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                              nonlinearity='relu')
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            # For multi-layer, PyTorch RNN supports residual via manual implementation
            # For simplicity, use standard RNN with residual in FC layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                              nonlinearity='relu', dropout=dropout)
            self.input_proj = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # FC layers with residual
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # RNN forward
        rnn_out, h = self.rnn(x)
        
        # For single layer, add input projection to hidden state
        if self.num_layers == 1:
            x_mean = x.mean(dim=1)
            h_residual = h[-1] + self.input_proj(x_mean)
        else:
            h_residual = h[-1]
        
        # FC with residual connection
        h1 = self.relu(self.fc1(self.dropout(h_residual)))
        h2 = h1 + h_residual  # Residual connection
        
        return self.fc2(h2)


class DeepResidualRNN(nn.Module):
    """Multi-layer RNN with residual connections between layers.
    
    Each layer has: h_l = RNN_l(h_{l-1}) + h_{l-1}
    Requires projection if hidden sizes differ.
    """
    
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        
        if num_layers < 2:
            raise ValueError("DeepResidualRNN requires num_layers >= 2")
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # First layer: input_size -> hidden_size
        self.rnn_layers = nn.ModuleList([
            nn.RNN(input_size, hidden_size, 1, batch_first=True, nonlinearity='relu')
        ])
        
        # Remaining layers: hidden_size -> hidden_size (with residual)
        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                nn.RNN(hidden_size, hidden_size, 1, batch_first=True, nonlinearity='relu')
            )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # First layer (no residual, different input size)
        h = x
        out, _ = self.rnn_layers[0](h)
        h = out[:, -1, :]  # Take last timestep
        
        # Subsequent layers with residual
        for layer in self.rnn_layers[1:]:
            # Expand h for sequence input
            h_seq = h.unsqueeze(1).expand(-1, x.size(1), -1)
            out, _ = layer(h_seq)
            h_new = out[:, -1, :]
            h = h_new + h  # Residual connection
        
        return self.fc(self.dropout(h))
