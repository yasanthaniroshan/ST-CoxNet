import torch
import torch.nn as nn
import torch.nn.functional as F

class WeibullHead(nn.Module):
    def __init__(self, context_dim:int,latent_dim:int):
        super().__init__()
        input_dim = context_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.lambda_head = nn.Linear(64, 1)
        self.k_head = nn.Linear(64, 1)

        nn.init.constant_(self.lambda_head.weight, 0.5)
        nn.init.constant_(self.k_head.weight, 0.5)

    def forward(self, c, z):
        x = torch.cat([c, z], dim=1)
        h = self.net(x)
        lambda_ = F.softplus(self.lambda_head(h)) + 1e-6
        k = F.softplus(self.k_head(h)) + 1e-6
        return lambda_, k