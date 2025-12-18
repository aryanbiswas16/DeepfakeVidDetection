from __future__ import annotations
import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
