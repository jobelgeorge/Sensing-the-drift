import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU encoder with temporal robustness for imbalanced regression.
    
    Architecture:
    - LayerNorm on input (temporal stability)
    - Conditional dropout (skip if dropout=0.0)
    - Clean separation: backbone → penultimate features → head
    - FDS-compatible (optional feature extraction)
    """
    
    def __init__(self, input_size, output_size, rnn_units=64, fc_units=64, 
                 dropout=[0.35, 0.25, 0.15], num_layers=2, use_batch_norm=False, use_input_norm=True):
        super(GRUModel, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        self.feat_dim = fc_units // 2
        self.use_input_norm = use_input_norm
        
        if isinstance(dropout, list):
            dropout_rnn = dropout[0] if len(dropout) > 0 else 0.0
            dropout_fc1 = dropout[1] if len(dropout) > 1 else 0.0
            dropout_fc2 = dropout[2] if len(dropout) > 2 else 0.0
        else:
            dropout_rnn = float(dropout)
            dropout_fc1 = dropout_rnn * 0.7 if dropout_rnn > 0 else 0.0
            dropout_fc2 = dropout_rnn * 0.5 if dropout_rnn > 0 else 0.0
        
        self.dropout_rnn = dropout_rnn
        self.dropout_fc1 = dropout_fc1
        self.dropout_fc2 = dropout_fc2
        
        if use_input_norm:
            self.input_norm = nn.LayerNorm(input_size)
        
        self.gru = nn.GRU(
            input_size, rnn_units, num_layers,
            batch_first=True,
            dropout=dropout_rnn if num_layers > 1 and dropout_rnn > 0 else 0,
            bidirectional=False
        )
        
        if dropout_rnn > 0:
            self.dropout1 = nn.Dropout(dropout_rnn)
        
        self.fc1 = nn.Linear(rnn_units, fc_units)
        
        if dropout_fc1 > 0:
            self.dropout2 = nn.Dropout(dropout_fc1)
        
        self.fc2 = nn.Linear(fc_units, fc_units // 2)
        
        self.activation = nn.ReLU()
        self.head = nn.Linear(fc_units // 2, output_size)
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature extraction for FDS.
        
        Args:
            x: input sequences (B, T, F)
            return_features: if True, return (features, predictions); else just predictions
        
        Returns:
            If return_features=True: (penultimate_features, predictions)
            If return_features=False: predictions only (default)
        """
        if self.use_input_norm:
            x = self.input_norm(x)
        
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        h = self.dropout1(last_output) if self.dropout_rnn > 0 else last_output
        h = self.activation(self.fc1(h))
        h = self.dropout2(h) if self.dropout_fc1 > 0 else h
        penult = self.activation(self.fc2(h))
        
        pred = self.head(penult)
        
        if return_features:
            return penult, pred
        else:
            return pred
