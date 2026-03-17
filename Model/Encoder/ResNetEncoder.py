import torch
import torch.nn as nn
from Model.ResNet import ResNetBlock1D

class Encoder(nn.Module):
    def __init__(self, latent_dim:int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11,stride=1,padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            ResNetBlock1D(16, 32, kernel_size=7),
            ResNetBlock1D(32, 64, kernel_size=5)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, rr_window)->torch.Tensor:
        """
        rr_window: [B, N]
        output:    [B, latent_dim]
        """
        x = rr_window.unsqueeze(1)
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        z = self.proj(h)
        return z