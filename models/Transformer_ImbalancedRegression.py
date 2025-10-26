import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Add positional information to input embeddings."""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    """
    Transformer for imbalanced time-series regression.

    Args:
        input_size: features per timestep
        d_model: hidden dim (128 recommended)
        nhead: attention heads (8 recommended)
        num_layers: transformer layers (3 recommended)
        dim_feedforward: FFN dim (512 recommended)
        dropout: 0.2 recommended
        max_seq_len: safety cap for PE
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 100
    ):
        super().__init__()

        self.d_model = d_model
        self.input_size = input_size

        # Input projection + (pre)norm improves stability
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm stabilizes deep encoders
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # MLP head (features after fc1 are used by FDS)
        self.fc1 = nn.Linear(d_model, 128)
        self.fc1_norm = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.fc2_norm = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(64, 1)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, x, return_features: bool = False):
        """
        x: [B, T, F]
        return_features: if True, also return the 128-dim features (for FDS)
        """
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x)  # [B, T, d_model]

        # Mean pool across time (empirically better for regression)
        token = encoded.mean(dim=1)  # [B, d_model]

        h = self.act(self.fc1(token))
        h = self.fc1_norm(h)
        features = h.clone()

        h = self.dropout1(h)
        h = self.act(self.fc2(h))
        h = self.fc2_norm(h)
        h = self.dropout2(h)

        pred = self.fc_out(h)

        if return_features:
            return pred, features
        return pred


__all__ = ['TransformerRegressor']
