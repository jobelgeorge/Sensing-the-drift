import torch.nn as nn
import torch

class UVOTELSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_experts=3):
        super(UVOTELSTM, self).__init__()
        # Using LSTM with data has sequence_length=20
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Expert heads: Each predicts [Mean, Log_Variance]
        # We predict for the last time step in the sequence
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] # [batch, hidden_dim]
        
        # Returns [num_experts, batch, 2]
        return torch.stack([expert(last_time_step) for expert in self.experts])