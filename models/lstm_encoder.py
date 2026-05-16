from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    # 返回最后一层的隐状态作为轨迹表示。
    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        x = self.input_fc(traj)
        _, (h, _) = self.lstm(x)
        return h[-1]
