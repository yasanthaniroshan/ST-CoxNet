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

        # 1. Encode all windows in the batch
        # Output z_seq: [B, T, latent_dim]
        z_seq = self.encoder(rr_windows.view(-1, window_size)).view(B, T, -1)
        z_seq = F.normalize(z_seq, dim=-1) # Normalize early for stability

        # 2. Extract Context (c_t) using the Autoregressive block (GRU)
        # Output c_seq: [B, T, context_dim]
        c_seq = self.ar_block(z_seq)

        total_loss = 0
        total_accuracy = 0
        count = 0

        # 3. Prediction Loop
        # Range ends early to ensure we have a 'k' step future ground truth available
        for t in range(T - self.number_of_prediction_steps):
            c_t = c_seq[:, t, :] # Context at current time step for the whole batch
            
            for k in range(1, self.number_of_prediction_steps + 1):
                # Predict the future latent for step k
                pred_k = self.Wk[k-1](c_t) # [B, latent_dim]
                # pred_k = self.Wk(c_t) # [B, latent_dim]
                pred_k = F.normalize(pred_k, dim=-1)

                # GROUND TRUTH: The actual future latent at step t+k
                # z_target_k: [B, latent_dim]
                z_target_k = z_seq[:, t + k, :] 
                z_target_k = F.normalize(z_target_k, dim=-1)

                # BATCH-WISE CONTRAST: 
                # We compute dot products between all B predictions and all B targets.
                # Logits shape: [B, B]
                # logits[i, j] is how well Sample i's prediction matches Sample j's actual future.
                logits = torch.matmul(pred_k, z_target_k.transpose(0, 1)) 
                logits /= self.temperature
                
                # The correct match is Sample i predicting Sample i (the diagonal)
                # labels: [0, 1, 2, ..., B-1]
                labels = torch.arange(B, device=rr_windows.device)
                
                # Cross Entropy treats this as a B-way classification task
                loss = F.cross_entropy(logits, labels)
                total_loss += loss
                
                # Accuracy: Did Sample i pick itself out of the batch?
                with torch.no_grad():
                    pred_class = torch.argmax(logits, dim=1)
                    correct_task = (pred_class == labels).float().mean().item()
                    total_accuracy += correct_task
                
                count += 1

        avg_loss = total_loss / count
        avg_accuracy = (total_accuracy / count) * 100 # Percentage for easier logging
        return avg_loss, avg_accuracy, z_seq, c_seq