import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(self, context_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.LayerNorm(context_dim // 2),
            nn.ReLU(),
            nn.Linear(context_dim // 2, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
