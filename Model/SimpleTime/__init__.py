import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTimeEncoder(nn.Module):
    """
    Encodes RR segments [B, T, W] into:
    - latent vector: [B, latent_dim] (for downstream time prediction)
    - projection vector: [B, proj_dim] (for contrastive optimization)
    """

    def __init__(self, window_size: int, latent_dim: int = 128, proj_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.window_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.sequence_encoder = nn.GRU(
            input_size=64,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, proj_dim),
        )
        self.window_size = window_size

    def forward(self, rr_segments: torch.Tensor):
        b, t, w = rr_segments.shape
        if w != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {w}")

        x = rr_segments.reshape(b * t, 1, w)
        x = self.window_encoder(x).squeeze(-1)  # [B*T, 64]
        x = x.view(b, t, -1)  # [B, T, 64]
        _, h = self.sequence_encoder(x)  # h: [num_layers, B, latent_dim]
        latent = h[-1]  # last layer hidden: [B, latent_dim]

        projection = self.projection_head(latent)
        projection = F.normalize(projection, dim=-1)
        return latent, projection


class TimeToEventHead(nn.Module):
    """
    Predicts normalized time-to-event from latent.
    No Sigmoid — output is unconstrained, trained with MSE on [0,1] targets.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(1)
