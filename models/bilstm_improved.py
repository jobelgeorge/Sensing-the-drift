import torch
import torch.nn as nn


class BiLSTMImproved(nn.Module):
    """
    Improved BiLSTM model with:
    - Bidirectional LSTM for better temporal modeling
    - LayerNorm instead of BatchNorm (more stable)
    - Deeper FC network with residual connections
    - Optimized dropout schedule
    """
    
    def __init__(self, input_size, output_size, rnn_units=128, fc_units=128, 
                 dropout=[0.3, 0.25, 0.2], num_layers=2, use_residual=True):
        super(BiLSTMImproved, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Handle dropout as list or single value
        if isinstance(dropout, list):
            dropout_rnn = dropout[0] if len(dropout) > 0 else 0.3
            dropout_fc1 = dropout[1] if len(dropout) > 1 else 0.25
            dropout_fc2 = dropout[2] if len(dropout) > 2 else 0.2
        else:
            dropout_rnn = dropout
            dropout_fc1 = dropout * 0.8
            dropout_fc2 = dropout * 0.6
        
        # Bidirectional LSTM - captures both forward and backward temporal patterns
        self.lstm = nn.LSTM(
            input_size, 
            rnn_units, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_rnn if num_layers > 1 else 0, 
            bidirectional=True  # ✅ KEY IMPROVEMENT
        )
        
        # LayerNorm for stability (better than BatchNorm for sequences)
        self.layer_norm = nn.LayerNorm(rnn_units * 2)  # *2 for bidirectional
        
        # Deeper FC network with residual connections
        # Output from BiLSTM: rnn_units * 2 (concat of forward and backward)
        self.fc1 = nn.Linear(rnn_units * 2, fc_units * 2)
        self.dropout1 = nn.Dropout(dropout_rnn)
        
        # Residual block (optional)
        if use_residual:
            self.fc_residual = nn.Linear(fc_units * 2, fc_units * 2)
            self.dropout_residual = nn.Dropout(dropout_fc1)
        
        self.fc2 = nn.Linear(fc_units * 2, fc_units)
        self.dropout2 = nn.Dropout(dropout_fc1)
        
        self.fc3 = nn.Linear(fc_units, fc_units // 2)
        self.dropout3 = nn.Dropout(dropout_fc2)
        
        self.fc4 = nn.Linear(fc_units // 2, output_size)
        
        self.activation = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:
                # Only initialize 2D+ tensors (not biases or 1D params)
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        # BiLSTM forward pass
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, rnn_units * 2)
        
        # Use last timestep output (contains info from both directions)
        last_output = lstm_out[:, -1, :]  # (batch, rnn_units * 2)
        
        # Apply layer normalization
        last_output = self.layer_norm(last_output)
        
        # First FC layer
        out = self.fc1(last_output)
        out = self.activation(out)
        out = self.dropout1(out)
        
        # Residual block (optional)
        if self.use_residual:
            identity = out
            out = self.fc_residual(out)
            out = self.activation(out)
            out = out + identity  # Residual connection
            out = self.dropout_residual(out)
        
        # Remaining FC layers
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.activation(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        
        return out


class BiLSTMWithAttention(nn.Module):
    """
    Advanced BiLSTM with self-attention mechanism
    Use this if BiLSTMImproved doesn't achieve target performance
    """
    
    def __init__(self, input_size, output_size, rnn_units=128, fc_units=128, 
                 dropout=0.3, num_layers=2, num_attention_heads=4):
        super(BiLSTMWithAttention, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, 
            rnn_units, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0, 
            bidirectional=True
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_units * 2,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(rnn_units * 2)
        self.layer_norm2 = nn.LayerNorm(rnn_units * 2)
        
        # FC layers
        self.fc1 = nn.Linear(rnn_units * 2, fc_units)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(fc_units, fc_units // 2)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        
        self.fc3 = nn.Linear(fc_units // 2, output_size)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # BiLSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, rnn_units * 2)
        
        # Apply self-attention over sequence
        # Query, Key, Value are all from lstm_out
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection + layer norm
        lstm_out = self.layer_norm1(lstm_out + attn_out)
        
        # Use last timestep after attention
        last_output = lstm_out[:, -1, :]
        last_output = self.layer_norm2(last_output)
        
        # FC layers
        out = self.fc1(last_output)
        out = self.activation(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out


if __name__ == "__main__":
    # Test the models
    batch_size = 32
    seq_len = 30
    input_size = 22
    
    # Test BiLSTMImproved
    model1 = BiLSTMImproved(input_size=input_size, output_size=1)
    x = torch.randn(batch_size, seq_len, input_size)
    y = model1(x)
    print(f"BiLSTMImproved output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Test BiLSTMWithAttention
    model2 = BiLSTMWithAttention(input_size=input_size, output_size=1)
    y = model2(x)
    print(f"\nBiLSTMWithAttention output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model2.parameters()):,}")
