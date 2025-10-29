import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """
    Production-ready BiLSTM with Attention for Imbalanced Regression.
    
    Key improvements:
    - Bidirectional LSTM for richer temporal features
    - Attention mechanism focuses on critical drift moments
    - Feature extraction layer (128-dim) for FDS compatibility
    - Layer normalization (stable with small batches)
    - Progressive dropout for regularization
    """
    
    def __init__(self, input_size, output_size=1, rnn_units=128, fc_units=128, 
                 dropout=0.3, num_layers=2, use_batch_norm=True, 
                 bidirectional=True, feature_dim=128):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = rnn_units
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.feature_dim = feature_dim
        
        # Handle dropout as list or single value
        if isinstance(dropout, list):
            dropout_rnn = dropout[0] if len(dropout) > 0 else 0.3
            dropout_fc1 = dropout[1] if len(dropout) > 1 else 0.25
            dropout_fc2 = dropout[2] if len(dropout) > 2 else 0.2
        else:
            dropout_rnn = dropout
            dropout_fc1 = dropout * 0.8
            dropout_fc2 = dropout * 0.6
        
        # Input normalization (LayerNorm > BatchNorm for stability)
        self.input_norm = nn.LayerNorm(input_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, rnn_units, num_layers, 
            batch_first=True, 
            dropout=dropout_rnn if num_layers > 1 else 0, 
            bidirectional=self.bidirectional
        )
        
        # Attention mechanism
        attn_input_size = self.hidden_size * self.num_directions
        self.attention_w = nn.Linear(attn_input_size, attn_input_size)
        self.attention_v = nn.Linear(attn_input_size, 1, bias=False)
        
        # Layer normalization after attention
        self.lstm_norm = nn.LayerNorm(attn_input_size)
        
        # Feature extraction head (for FDS)
        self.feature_head = nn.Sequential(
            nn.Linear(attn_input_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_fc1 * 0.5)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_fc1),
            nn.Linear(feature_dim, fc_units),
            nn.LayerNorm(fc_units),
            nn.GELU(),
            nn.Dropout(dropout_fc2),
            nn.Linear(fc_units, fc_units // 2),
            nn.LayerNorm(fc_units // 2),
            nn.GELU(),
            nn.Dropout(dropout_fc2 * 0.5),
            nn.Linear(fc_units // 2, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Proper initialization for LSTM + Linear layers"""
        for name, param in self.named_parameters():
            if 'lstm' in name and 'weight' in name:
                if param.dim() >= 2:
                    nn.init.orthogonal_(param)
            elif 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, T, F] - batch, time, features
            return_features: Return 128-dim features for FDS
        
        Returns:
            pred: [B, 1]
            features: [B, 128] (if return_features=True)
        """
        # Input normalization
        x = self.input_norm(x)
        
        # BiLSTM forward pass
        lstm_out, _ = self.lstm(x)  # [B, T, hidden*2]
        
        # Attention mechanism
        attn_scores = torch.tanh(self.attention_w(lstm_out))  # [B, T, hidden*2]
        attn_weights = F.softmax(self.attention_v(attn_scores), dim=1)  # [B, T, 1]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # [B, hidden*2]
        
        # Layer normalization
        context_vector = self.lstm_norm(context_vector)
        
        # Feature extraction (for FDS)
        features = self.feature_head(context_vector)  # [B, 128]
        
        # Regression
        pred = self.regressor(features)  # [B, 1]
        
        if return_features:
            return pred, features
        return pred