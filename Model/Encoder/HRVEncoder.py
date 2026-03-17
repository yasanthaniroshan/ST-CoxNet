import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(32, 64),
            nn.GELU(),

            nn.Linear(64, latent_dim)
        )

    def forward(self, hrv):
        """
        hrv: [B, input_dim]
        output: [B, latent_dim]
        """
        z = self.net(hrv)
        return z