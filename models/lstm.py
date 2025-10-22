import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Enhanced LSTM model with stronger regularization (No Attention)"""
    
    def __init__(self, input_size, output_size, rnn_units=64, fc_units=64, 
                 dropout=[0.5, 0.35, 0.25], num_layers=2, use_batch_norm=True):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        # Handle dropout as list or single value
        if isinstance(dropout, list):
            dropout_rnn = dropout[0] if len(dropout) > 0 else 0.5
            dropout_fc1 = dropout[1] if len(dropout) > 1 else 0.35
            dropout_fc2 = dropout[2] if len(dropout) > 2 else 0.25
        else:
            dropout_rnn = dropout
            dropout_fc1 = dropout * 0.7
            dropout_fc2 = dropout * 0.5
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM Layer with dropout
        self.lstm = nn.LSTM(input_size, rnn_units, num_layers, 
                           batch_first=True, dropout=dropout_rnn if num_layers > 1 else 0, 
                           bidirectional=False)
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(rnn_units)
        
        # Output layers with stronger regularization
        self.dropout1 = nn.Dropout(dropout_rnn)
        self.fc1 = nn.Linear(rnn_units, fc_units)
        self.dropout2 = nn.Dropout(dropout_fc1)
        self.fc2 = nn.Linear(fc_units, fc_units // 2)
        self.dropout3 = nn.Dropout(dropout_fc2)
        self.fc3 = nn.Linear(fc_units // 2, output_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Apply batch normalization
        if self.use_batch_norm:
            last_output = self.batch_norm(last_output)
        
        # Dense layers with progressive regularization
        out = self.dropout1(last_output)
        out = self.activation(self.fc1(out))
        out = self.dropout2(out)
        out = self.activation(self.fc2(out))
        out = self.dropout3(out)
        out = self.fc3(out)
        
        return out
