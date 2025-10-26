import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Add positional information to input embeddings.
    Critical for Transformer to understand time ordering.
    """
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    """
    Transformer for time-series regression with imbalanced targets.
    
    Key advantages for imbalanced data:
    - Attention weights show WHICH timesteps matter for rare predictions
    - Can learn "warning signs" of extreme events
    - No sequential processing bottleneck
    
    Args:
        input_size: Number of input features per timestep
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward network dimension
        dropout: Dropout rate
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
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm: better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output head
        self.fc1 = nn.Linear(d_model, 128)
        self.fc1_norm = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc2_norm = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(64, 1)
        
        self.relu = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, return_features=False, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: [B, T, F] - (batch, time, features)
            return_features: Return intermediate features for FDS
            return_attention: Return attention weights for analysis
        
        Returns:
            predictions: [B, 1]
            (optional) features: [B, 128]
            (optional) attention_weights: [B, num_heads, T, T]
        """
        # Project input to d_model
        x = self.input_projection(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        if return_attention:
            # Extract attention weights (requires custom forward)
            encoded, attention_weights = self._forward_with_attention(x)
        else:
            encoded = self.transformer_encoder(x)  # [B, T, d_model]
            attention_weights = None
        
        # Use last timestep (or mean pooling for better stability)
        # Option 1: Last timestep
        out = encoded[:, -1, :]
        
        # Option 2: Mean pooling (often better for regression)
        # out = encoded.mean(dim=1)
        
        # FC layers
        out = self.relu(self.fc1(out))
        out = self.fc1_norm(out)
        features = out.clone()  # For FDS
        
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.fc2_norm(out)
        out = self.dropout2(out)
        predictions = self.fc_out(out)
        
        if return_features and return_attention:
            return predictions, features, attention_weights
        elif return_features:
            return predictions, features
        elif return_attention:
            return predictions, attention_weights
        return predictions
    
    def _forward_with_attention(self, x):
        """Forward pass that captures attention weights."""
        # This is simplified - full implementation requires modifying TransformerEncoder
        # For now, just return encoded output
        encoded = self.transformer_encoder(x)
        # In practice, you'd extract attention from each layer
        return encoded, None
    
    def analyze_attention(self, x):
        """
        Analyze which timesteps the model attends to for prediction.
        Useful for understanding "what triggers rare events".
        
        Args:
            x: [B, T, F] - Single batch
        
        Returns:
            attention_map: [num_heads, T] - Average attention to each timestep
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
            # Average across all heads and layers
            # attention_weights shape: [num_layers, B, num_heads, T, T]
            # We want: which input positions get high attention
            avg_attention = attention_weights.mean(dim=(0, 1, 2))  # [T]
            return avg_attention
