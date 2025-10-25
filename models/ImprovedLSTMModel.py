import torch
import torch.nn as nn


class ImprovedLSTMModel(nn.Module):
    """
    Enhanced LSTM for imbalanced regression with:
    1. Increased capacity for capturing rare dynamics
    2. Layer normalization (stable with small batch sizes)
    3. Residual connections to prevent gradient vanishing
    4. Multi-scale temporal feature extraction
    5. Uncertainty-aware prediction (optional)
    
    Args:
        input_size: Number of input features
        output_size: Number of output features (typically 1 for regression)
        rnn_units: LSTM hidden size (increase to 128-256 for complex patterns)
        num_layers: Number of stacked LSTM layers
        fc_units: Fully connected layer size
        dropout: Dropout rate (single value or list)
        use_residual: Add residual connections
        use_attention: Use attention over time steps
        predict_uncertainty: Output (mean, log_variance) for epistemic uncertainty
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        rnn_units: int = 128,  # Increased from 64
        num_layers: int = 2,
        fc_units: int = 128,
        dropout: float = 0.3,  # Reduced - less dropout for rare samples
        use_residual: bool = True,
        use_attention: bool = False,
        predict_uncertainty: bool = False
    ):
        super().__init__()
        
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.predict_uncertainty = predict_uncertainty
        
        # Input normalization (LayerNorm instead of BatchNorm for stability)
        self.input_norm = nn.LayerNorm(input_size)
        
        # Multi-layer LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size,
            rnn_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Layer normalization after LSTM (better than BatchNorm for sequences)
        self.lstm_norm = nn.LayerNorm(rnn_units)
        
        # Optional: Attention mechanism for focusing on critical time steps
        if use_attention:
            self.attention = TemporalAttention(rnn_units)
        
        # Residual projection if needed
        if use_residual and input_size != rnn_units:
            self.residual_proj = nn.Linear(input_size, rnn_units)
        elif use_residual:
            self.residual_proj = nn.Identity()
        
        # Progressive fully connected layers
        self.dropout1 = nn.Dropout(dropout * 0.8)
        self.fc1 = nn.Linear(rnn_units, fc_units)
        self.fc1_norm = nn.LayerNorm(fc_units)
        
        self.dropout2 = nn.Dropout(dropout * 0.6)
        self.fc2 = nn.Linear(fc_units, fc_units // 2)
        self.fc2_norm = nn.LayerNorm(fc_units // 2)
        
        self.dropout3 = nn.Dropout(dropout * 0.4)
        
        # Output layer
        if predict_uncertainty:
            # Output both mean and log-variance for uncertainty estimation
            self.fc_out = nn.Linear(fc_units // 2, output_size * 2)
        else:
            self.fc_out = nn.Linear(fc_units // 2, output_size)
        
        self.activation = nn.GELU()  # GELU often better than ReLU
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
       """Xavier/He initialization for better convergence."""
       for name, param in self.named_parameters():
           if 'weight' in name:
               if 'lstm' in name and param.dim() >= 2:
                   nn.init.orthogonal_(param)  # Orthogonal for RNN weight matrices only
               elif 'norm' in name:
                   nn.init.constant_(param, 1.0)  # LayerNorm weights = 1.0
               elif param.dim() >= 2:
                   nn.init.kaiming_normal_(param, nonlinearity='relu')  # Linear layers
           elif 'bias' in name:
               nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, features]
        
        Returns:
            If predict_uncertainty=False: [batch, output_size]
            If predict_uncertainty=True: (mean [batch, output_size], 
                                         log_var [batch, output_size])
        """
        # Input normalization
        x = self.input_norm(x)
        
        # Store for residual connection
        if self.use_residual:
            residual = self.residual_proj(x[:, -1, :])  # Project last timestep
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # [B, T, H]
        
        # Apply layer norm
        lstm_out = self.lstm_norm(lstm_out)
        
        # Attention or last output
        if self.use_attention:
            context = self.attention(lstm_out)  # [B, H]
        else:
            context = lstm_out[:, -1, :]  # Last timestep [B, H]
        
        # Residual connection
        if self.use_residual:
            context = context + residual
        
        # Fully connected layers with progressive dropout
        out = self.dropout1(context)
        out = self.activation(self.fc1(out))
        out = self.fc1_norm(out)
        
        out = self.dropout2(out)
        out = self.activation(self.fc2(out))
        out = self.fc2_norm(out)
        
        out = self.dropout3(out)
        out = self.fc_out(out)
        
        if self.predict_uncertainty:
            # Split output into mean and log-variance
            mean = out[:, 0]
            log_var = out[:, 1]
            return mean, log_var
        else:
            return out


class TemporalAttention(nn.Module):
    """
    Attention mechanism over time steps.
    Helps model focus on critical moments (e.g., sudden maneuvers).
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: [batch, seq_len, hidden]
        
        Returns:
            context: [batch, hidden] - weighted sum of time steps
        """
        # Compute attention scores
        scores = self.attention_weights(lstm_output)  # [B, T, 1]
        attention = torch.softmax(scores, dim=1)  # [B, T, 1]
        
        # Weighted sum
        context = (lstm_output * attention).sum(dim=1)  # [B, H]
        
        return context


class UncertaintyLoss(nn.Module):
    """
    Negative log-likelihood loss for uncertainty-aware predictions.
    
    The model predicts (mean, log_variance), and we penalize:
    - Prediction error weighted by confidence (low variance)
    - Overconfidence (variance too small)
    
    This helps the model be more conservative on few-shot samples where
    it should be uncertain.
    
    Formula: Loss = 0.5 * (exp(-log_var) * |pred - target|^2 + log_var)
    
    Intuition:
    - High confidence (low log_var) + large error → big penalty
    - Low confidence (high log_var) + large error → smaller penalty
    - Very low confidence everywhere → penalized by log_var term
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, mean, log_var, target):
        """
        Args:
            mean: Predicted mean [B]
            log_var: Predicted log-variance [B]
            target: Ground truth [B]
        
        Returns:
            Scalar loss
        """
        target = target.squeeze(-1)
        
        # Negative log-likelihood
        precision = torch.exp(-log_var)
        loss = 0.5 * (precision * (mean - target) ** 2 + log_var)
        
        return loss.mean()


# ============================================================================
# Multi-Scale Temporal CNN-LSTM Hybrid (Advanced Option)
# ============================================================================

class MultiScaleLSTM(nn.Module):
    """
    Combines CNN for local pattern extraction with LSTM for long-term dynamics.
    
    Architecture:
    1. Multi-scale 1D convolutions capture patterns at different time scales
    2. LSTM processes the enriched features
    3. Particularly effective for detecting rare transient events
    
    Use this if standard LSTM plateaus and you need more representation power.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        cnn_channels: int = 64,
        rnn_units: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_size)
        
        # Multi-scale 1D convolutions (different kernel sizes)
        self.conv_small = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(input_size, cnn_channels, kernel_size=5, padding=2)
        self.conv_large = nn.Conv1d(input_size, cnn_channels, kernel_size=7, padding=3)
        
        # Combine multi-scale features
        self.conv_fusion = nn.Conv1d(cnn_channels * 3, cnn_channels, kernel_size=1)
        self.conv_norm = nn.LayerNorm(cnn_channels)
        
        # LSTM processes fused features
        self.lstm = nn.LSTM(
            cnn_channels,
            rnn_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.lstm_norm = nn.LayerNorm(rnn_units)
        
        # Output head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_units, rnn_units // 2),
            nn.GELU(),
            nn.LayerNorm(rnn_units // 2),
            nn.Dropout(dropout * 0.5),
            nn.Linear(rnn_units // 2, output_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features]
        
        Returns:
            [batch, output_size]
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Transpose for Conv1d: [B, F, T]
        x = x.transpose(1, 2)
        
        # Multi-scale convolutions
        x_small = torch.relu(self.conv_small(x))
        x_medium = torch.relu(self.conv_medium(x))
        x_large = torch.relu(self.conv_large(x))
        
        # Concatenate and fuse
        x_multi = torch.cat([x_small, x_medium, x_large], dim=1)
        x_fused = torch.relu(self.conv_fusion(x_multi))
        
        # Transpose back: [B, T, C]
        x_fused = x_fused.transpose(1, 2)
        x_fused = self.conv_norm(x_fused)
        
        # LSTM
        lstm_out, _ = self.lstm(x_fused)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Use last output
        out = lstm_out[:, -1, :]
        
        # Final prediction
        out = self.fc(out)
        
        return out
