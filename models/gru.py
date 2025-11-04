import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU encoder with DIR-style feature extraction for FDS calibration.
    
    Architecture:
    - NO input normalization (removed to preserve variance)
    - NO BatchNorm (removed to preserve scale)
    - Minimal dropout (only in backbone, not after FDS)
    - Clean separation: backbone → penultimate features → head
    - Head is applied AFTER FDS calibration
    """
    
    def __init__(self, input_size, output_size, rnn_units=64, fc_units=64, 
                 dropout=[0.0, 0.0, 0.0], num_layers=2, use_batch_norm=False):
        super(GRUModel, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        self.feat_dim = fc_units // 2  # Penultimate feature dimension for FDS
        
        # Handle dropout as list or single value
        if isinstance(dropout, list):
            dropout_rnn = dropout[0] if len(dropout) > 0 else 0.0
            dropout_fc1 = dropout[1] if len(dropout) > 1 else 0.0
            dropout_fc2 = dropout[2] if len(dropout) > 2 else 0.0
        else:
            dropout_rnn = float(dropout)
            dropout_fc1 = dropout_rnn * 0.7
            dropout_fc2 = dropout_rnn * 0.5
        
        # NO input normalization - preserve raw feature distribution for FDS
        
        # GRU backbone
        self.gru = nn.GRU(
            input_size, rnn_units, num_layers,
            batch_first=True,
            dropout=dropout_rnn if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # NO BatchNorm - preserve feature scale for FDS
        
        # Backbone MLP to generate penultimate features
        self.dropout1 = nn.Dropout(dropout_rnn)
        self.fc1 = nn.Linear(rnn_units, fc_units)
        self.dropout2 = nn.Dropout(dropout_fc1)
        self.fc2 = nn.Linear(fc_units, fc_units // 2)  # Penultimate layer
        
        self.activation = nn.ReLU()
        
        # Final head (applied AFTER FDS calibration)
        # NO dropout/norm after this point
        self.head = nn.Linear(fc_units // 2, output_size)
    
    def forward(self, x, return_features=True):
        """
        Forward pass with optional feature extraction for FDS.
        
        Args:
            x: input sequences (B, T, F)
            return_features: if True, return (features, predictions); else just predictions
        
        Returns:
            If return_features=True: (penultimate_features, predictions)
            If return_features=False: predictions only
        """
        # GRU forward (no input norm)
        gru_out, _ = self.gru(x)
        
        # Use last timestep output
        last_output = gru_out[:, -1, :]  # (B, rnn_units)
        
        # Backbone MLP → penultimate features
        h = self.dropout1(last_output)
        h = self.activation(self.fc1(h))
        h = self.dropout2(h)
        penult = self.activation(self.fc2(h))  # (B, feat_dim) ← FDS operates here
        
        # Head (applied after FDS in training loop)
        pred = self.head(penult)
        
        if return_features:
            return penult, pred
        else:
            return pred
