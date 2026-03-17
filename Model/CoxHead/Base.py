from torch import nn
import torch
class CoxHead(nn.Module):
    def __init__(self, context_dim:int,latent_dim:int,dropout:float=0.2):
        super().__init__()
        input_dim = context_dim + latent_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, c, z):
        context = self.context_norm(c)
        latent = self.latent_norm(z)
        combined = torch.cat([context, latent], dim=1)
        return self.net(combined)
