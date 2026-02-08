import torch.nn as nn
import torch

class VanillaRNN_UVOTE_FDS(nn.Module):
    """Vanilla RNN + U-Vote + FDS"""

    def __init__(self, input_size=22, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu',
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shared feature layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Two heads
        self.mu_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)
        
        # FDS module (will be initialized externally)
        self.fds = None
        self.current_epoch = 0

    def forward(self, x, y=None, epoch=None):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor (B, T, F)
        y : torch.Tensor (B, 1), optional - needed for FDS
        epoch : int, optional - needed for FDS
        """
        _, h = self.rnn(x)
        h = h[-1]
        h = self.dropout(h)
        
        features = self.fc(h)  # (B, hidden_size)
        
        # Apply FDS if available and training
        if self.fds is not None and y is not None and epoch is not None:
            features = self.fds(features, y, epoch)
        
        mu = self.mu_head(features)
        log_var = self.logvar_head(features)

        return mu, log_var