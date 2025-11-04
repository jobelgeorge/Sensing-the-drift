import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    RNN model for sideslip angle prediction with FDS support.
    Returns (features, predictions) for DIR-style training.
    """
    
    def __init__(self, input_size, output_size, rnn_units=64, fc_units=64, 
                 dropout=[0.0, 0.0, 0.0], num_layers=2, use_batch_norm=False):
        super(RNNModel, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        self.feat_dim = fc_units // 2  # Feature dimension for FDS
        
        # RNN Layer - NO dropout
        self.rnn = nn.RNN(input_size, rnn_units, num_layers, 
                         batch_first=True, dropout=0.0, 
                         nonlinearity='relu', bidirectional=False)
        
        # Backbone (penultimate features)
        self.fc1 = nn.Linear(rnn_units, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units // 2)
        
        # Head (final prediction layer)
        self.head = nn.Linear(fc_units // 2, output_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x, return_features=True):
        """
        Forward pass with optional feature extraction for FDS.
        
        Args:
            x: input sequences (B, T, F)
            return_features: if True, return (features, predictions); else just predictions
        
        Returns:
            If return_features=True: (features, predictions)
            If return_features=False: predictions only
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Use last timestep output
        last_output = rnn_out[:, -1, :]
        
        # Backbone - generate penultimate features
        feat = self.activation(self.fc1(last_output))
        feat = self.activation(self.fc2(feat))  # This is the penultimate layer
        
        # Head - final prediction
        pred = self.head(feat)
        
        if return_features:
            return feat, pred
        else:
            return pred
