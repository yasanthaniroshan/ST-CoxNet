import torch.nn as nn
import torch
from Model.AutoregressiveBlock import ARBlock
from Model.PredictionHead.HRVPredictor.MultiStepPredictor import MultiStepHRVPredictor
from Model.Encoder.RREncoder import Encoder
from Metadata import CPCPreModelConfig

class CPCPreModel(nn.Module):
    def __init__(self, config:CPCPreModelConfig):
        super().__init__()
        self.encoder = Encoder(latent_dim=config.encoder.latent_dim)
        self.context = ARBlock(latent_dim=config.ar.latent_dim, context_dim=config.ar.context_dim)
        self.predictor = MultiStepHRVPredictor(context_dim=config.predictor.context_dim, num_targets=config.predictor.num_targets, num_heads=config.predictor.num_heads)

    def forward(self, rr_windows: torch.Tensor) -> torch.Tensor:
        """
        rr_windows: [B, T, W] 
        Returns:
            c_seq: [B, T, CONTEXT_SIZE]
        """
        
        B, T, W = rr_windows.shape
        z_list = []

        for t in range(T):
            z_t = self.encoder(rr_windows[:, t, :])  # [B, LATENT_SIZE]
            z_list.append(z_t)

        z_seq = torch.stack(z_list, dim=1)  # [B, T, LATENT_SIZE]
        c_seq = self.context(z_seq)         # [B, T, CONTEXT_SIZE]
        return c_seq



