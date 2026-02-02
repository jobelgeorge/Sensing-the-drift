"""BiLSTM with Multi-head Attention for architecture comparison."""

import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with multi-head attention."""
    
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, num_heads=4, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attention(out, out, out)
        out = self.norm(out + attn_out)
        return self.fc(self.dropout(out[:, -1, :]))
