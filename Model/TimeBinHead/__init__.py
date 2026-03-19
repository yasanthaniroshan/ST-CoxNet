import torch
import torch.nn as nn


class TimeBinHead(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        input_dim = context_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, context: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, embedding], dim=1)
        return self.net(x)
