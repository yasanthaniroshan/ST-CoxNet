import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Encoder.BaseEncoder import Encoder
from Model.AutoregressiveBlock import ARBlock
from Model.PredictionHead import PredictionHead

class CPC(nn.Module):
    def __init__(self, latent_dim:int, context_dim:int, number_of_prediction_steps:int,temperature:float=0.1,dropout:float=0.1):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout=dropout)
        self.ar_block = ARBlock(latent_dim, context_dim)
        self.number_of_prediction_steps = number_of_prediction_steps
        self.temperature = temperature
        self.Wk = nn.ModuleList([nn.Linear(context_dim, latent_dim) for _ in range(number_of_prediction_steps)])
        # self.Wk = nn.Linear(context_dim, latent_dim)


    def forward(self, rr_windows: torch.Tensor) -> torch.Tensor:
        B, T, window_size = rr_windows.shape

        # 1. Encode all windows → z_seq: [B, T, latent_dim]
        z_seq = self.encoder(rr_windows.view(-1, window_size)).view(B, T, -1)
        z_seq = F.normalize(z_seq, dim=-1)

        # 2. Autoregressive context → c_seq: [B, T, context_dim]
        c_seq = self.ar_block(z_seq)

        total_loss = 0.0
        total_accuracy = 0.0
        count = 0

        # 3. Sequence-wise Contrastive Loop
        for t in range(T - self.number_of_prediction_steps):
            c_t = c_seq[:, t, :]  # [B, context_dim]

            for k in range(1, self.number_of_prediction_steps + 1):
                # Prediction from context at t → [B, latent_dim]
                pred_k = F.normalize(self.Wk[k-1](c_t), dim=-1)

                # Positive: actual future latent at t+k → [B, latent_dim]
                z_positive = z_seq[:, t + k, :]  # [B, latent_dim]

                # Negatives: all OTHER timesteps in the sequence EXCEPT t+k
                # neg_indices: all time steps excluding t+k → [T-1] indices
                neg_indices = [i for i in range(T) if i != (t + k)]
                z_negatives = z_seq[:, neg_indices, :]  # [B, T-1, latent_dim]

                # --- Compute logits sequence-wise (per sample in batch) ---
                # positive_logit: [B, 1]
                positive_logit = torch.sum(pred_k * z_positive, dim=-1, keepdim=True) / self.temperature

                # negative_logits: [B, T-1]
                # pred_k[:, None, :] → [B, 1, latent_dim]
                # z_negatives       → [B, T-1, latent_dim]
                negative_logits = torch.bmm(
                    pred_k.unsqueeze(1),        # [B, 1, latent_dim]
                    z_negatives.transpose(1, 2) # [B, latent_dim, T-1]
                ).squeeze(1) / self.temperature  # [B, T-1]

                # Concatenate: positive is always at index 0 → [B, T]
                logits = torch.cat([positive_logit, negative_logits], dim=1)  # [B, T]

                # Labels: correct answer is always index 0 (the positive)
                labels = torch.zeros(B, dtype=torch.long, device=rr_windows.device)

                loss = F.cross_entropy(logits, labels)
                total_loss += loss

                with torch.no_grad():
                    pred_class = torch.argmax(logits, dim=1)
                    correct = (pred_class == labels).float().mean().item()
                    total_accuracy += correct

                count += 1

        avg_loss = total_loss / count
        avg_accuracy = (total_accuracy / count) * 100
        return avg_loss, avg_accuracy, z_seq, c_seq