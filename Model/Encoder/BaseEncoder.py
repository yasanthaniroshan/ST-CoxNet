import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim:int,window_size:int,dropout:float=0.1):
        super().__init__()
        # Convolutional feature extractor for a single RR window [B, 1, window_size]
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, padding=5,stride=2), # [B, 1, window_size] -> [B, 8, window_size/2]
            nn.GroupNorm(2,8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(8, 16, kernel_size=9, padding=5,stride=2), # [B, 8, window_size/2] -> [B, 16, window_size/4]
            nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=5, padding=2,stride=2), # [B, 16, window_size/4] -> [B, 32, window_size/8]
            nn.GroupNorm(8,32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Infer exact flattened size from the real conv stack.
        # Using window_size//8 is wrong here because stride/padding rounds up.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, window_size)
            flat_dim = self.feature_extractor(dummy).flatten(1).shape[1]

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        """
        x: [B, 1, window_size]
        output: [B, latent_dim]
        """
        x = x.unsqueeze(1)  # [B, 1, window_size]
        x = self.feature_extractor(x)
        z = self.projection(x)     # [B, latent_dim]
        return z