import torch
import torch.nn as nn
import torch.nn.functional as F  # <-- Make sure to import functional

class LSTMModel(nn.Module):
    """
    Enhanced LSTM model with Bidirectional layer, Attention mechanism, 
    and stronger regularization.
    """
    
    def __init__(self, input_size, output_size, rnn_units=64, fc_units=64, 
                 dropout=[0.5, 0.35, 0.25], num_layers=2, use_batch_norm=True, 
                 bidirectional=True): # <-- Add bidirectional argument
        super(LSTMModel, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1 # Used for calculations
        
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
                           bidirectional=self.bidirectional) # <-- Use bidirectional
        
        # --- ATTENTION MECHANISM LAYERS ---
        # The input to the attention mechanism will be the output of the LSTM
        # which is (rnn_units * num_directions)
        attn_input_size = self.hidden_size * self.num_directions
        self.attention_w = nn.Linear(attn_input_size, attn_input_size)
        self.attention_v = nn.Linear(attn_input_size, 1, bias=False)
        # --- END ATTENTION ---

        # Batch normalization (input size adjusted for bidirectional)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(attn_input_size)
        
        # Output layers with stronger regularization (input size adjusted for bidirectional)
        self.dropout1 = nn.Dropout(dropout_rnn)
        self.fc1 = nn.Linear(attn_input_size, fc_units) 
        self.dropout2 = nn.Dropout(dropout_fc1)
        self.fc2 = nn.Linear(fc_units, fc_units // 2)
        self.dropout3 = nn.Dropout(dropout_fc2)
        self.fc3 = nn.Linear(fc_units // 2, output_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, rnn_units * num_directions)
        lstm_out, _ = self.lstm(x)
        
        # --- ATTENTION MECHANISM ---
        # Calculate attention scores
        # (batch_size, seq_len, rnn_units * num_directions)
        attn_scores = torch.tanh(self.attention_w(lstm_out))
        
        # Calculate attention weights
        # (batch_size, seq_len, 1)
        attn_weights = F.softmax(self.attention_v(attn_scores), dim=1)
        
        # Calculate context vector (weighted sum of hidden states)
        # (batch_size, rnn_units * num_directions)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Set last_output to the context vector from attention
        last_output = context_vector
        # --- END ATTENTION ---
        
        # Apply batch normalization
        if self.use_batch_norm:
            last_output = self.batch_norm(last_output)
        
        # Dense layers with progressive regularization
        out = self.dropout1(last_output)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout3(out)
        out = self.fc3(out)
        return out