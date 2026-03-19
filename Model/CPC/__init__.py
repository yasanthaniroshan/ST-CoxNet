import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Encoder.BaseEncoder import Encoder
from Model.AutoregressiveBlock import ARBlock

class CPC(nn.Module):
    def __init__(self, latent_dim:int, context_dim:int, number_of_prediction_steps:int,window_size:int,temperature:float=0.1,dropout:float=0.1):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, window_size=window_size, dropout=dropout)
        self.ar_block = ARBlock(latent_dim=latent_dim, context_dim=context_dim)
        self.number_of_prediction_steps = number_of_prediction_steps
        self.temperature = temperature
        self.Wk = nn.ModuleList([nn.Linear(context_dim, latent_dim) for _ in range(number_of_prediction_steps)])

    def forward(self, rr_windows: torch.Tensor) -> torch.Tensor:
        B, T, window_size = rr_windows.shape

        z_seq = self.encoder(rr_windows.view(-1, window_size)).view(B, T, -1)
        z_seq = F.normalize(z_seq, dim=-1)
        c_seq = self.ar_block(z_seq)

        total_loss = 0.0
        total_accuracy = 0.0
        count = 0

        for t in range(T - self.number_of_prediction_steps):
            c_t = c_seq[:, t, :]

            for k in range(1, self.number_of_prediction_steps + 1):
                pred_k = F.normalize(self.Wk[k - 1](c_t), dim=-1)  # [B, latent_dim]
                z_future = z_seq[:, t + k, :]  # [B, latent_dim]

                # Cross-batch InfoNCE: logits[i,j] = sim(pred_i, z_future_j)
                logits = torch.matmul(pred_k, z_future.t()) / self.temperature  # [B, B]
                labels = torch.arange(B, device=rr_windows.device)

                loss = F.cross_entropy(logits, labels)
                total_loss += loss

                with torch.no_grad():
                    correct = (logits.argmax(dim=1) == labels).float().mean().item()
                    total_accuracy += correct

                count += 1

        avg_loss = total_loss / count
        avg_accuracy = (total_accuracy / count) * 100
        return avg_loss, avg_accuracy, z_seq, c_seq