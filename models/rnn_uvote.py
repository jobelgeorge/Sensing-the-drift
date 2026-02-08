import torch
import torch.nn as nn

class VanillaRNN_UVOTE(nn.Module):
    """
    Vanilla RNN + U-Vote
    Predicts:
      - mu      (mean sideslip)
      - log_var (log variance = uncertainty)
    """

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

        # 🔹 Two heads (U-Vote)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.rnn(x)           # h: (num_layers, B, hidden)
        h = h[-1]                    # last layer → (B, hidden)

        h = self.dropout(h)
        features = self.fc(h)

        mu = self.mu_head(features)
        log_var = self.logvar_head(features)

        return mu, log_var
