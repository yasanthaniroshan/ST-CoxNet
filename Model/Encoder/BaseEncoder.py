
from torch import nn
import torch
import torch.nn.functional as F
from Model.SEBlock import SEBlock
from Model.ResNet import ResNetBlock1D

class Encoder(nn.Module):
    def __init__(self, latent_dim:int,dropout:float=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            ResNetBlock1D(1, 16, kernel_size=3),
            nn.Dropout(dropout)
        )
        self.branch2 = nn.Sequential(
            ResNetBlock1D(1, 16, kernel_size=5),
            nn.Dropout(dropout)
        )

        self.branch3 = nn.Sequential(
            ResNetBlock1D(1, 16, kernel_size=7),
            nn.Dropout(dropout)
        )
        self.se = SEBlock(16*3)

        self.proj = nn.Sequential(
            nn.Linear(16*3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim)
            )


    def forward(self, rr_window)->torch.Tensor:
        """
        rr_window: [B, window_size]
        output:    [B, LATENT_SIZE]
        """
        x = rr_window.unsqueeze(1)
        x1 = self.branch1(x)  # [B, 32, L]
        x2 = self.branch2(x)  # [B, 32, L]
        x3 = self.branch3(x)  # [B, 32, L]
        x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 96, L]
        f_atten = self.se(x_cat)  # [B, 96, L]
        h = F.adaptive_avg_pool1d(f_atten, 1).squeeze(-1)  # [B, 96]
        z = self.proj(h)
        return z

