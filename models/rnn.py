import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    Production-ready RNN for sequence regression.
    
    Architecture:
    - Multi-layer RNN with configurable hidden size and dropout
    - Two-layer MLP head with ReLU activation and dropout
    - Optional batch normalization for stability
    
    Args:
        input_size: Number of input features
        hidden_size: RNN hidden state size (default: 128)
        num_layers: Number of RNN layers (default: 2)
        dropout: Dropout probability (applied between RNN layers and in MLP)
        use_batch_norm: Whether to use batch normalization in MLP head
        output_size: Number of output values (default: 1 for regression)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = False,
        output_size: int = 1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        # RNN backbone with dropout between layers (if num_layers > 1)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity='tanh'  # tanh is standard for RNN (more stable than relu)
        )
        
        # MLP head: hidden_size -> hidden_size//2 -> output_size
        self.use_batch_norm = use_batch_norm
        
        layers = []
        layers.append(nn.Linear(hidden_size, hidden_size // 2))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size // 2, output_size))
        
        self.head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:  # Biases
                nn.init.zeros_(param)
            elif 'head' in name and 'weight' in name:  # MLP weights
                nn.init.kaiming_normal_(param, nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch_size, seq_length, input_size)
        
        Returns:
            predictions: Output predictions (batch_size, output_size)
        """
        # RNN forward: process entire sequence
        # rnn_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        rnn_out, h_n = self.rnn(x)
        
        # Use last timestep output (most relevant for prediction)
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden_size)
        
        # MLP head: project to output
        output = self.head(last_hidden)  # (batch, output_size)
        
        return output
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
