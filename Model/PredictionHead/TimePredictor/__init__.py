import torch
import torch.nn as nn

class TimePredictor(nn.Module):
    def __init__(self, embedding_dim:int, context_dim:int,dropout:float=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim + context_dim,
            hidden_size=32,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, embedding: torch.Tensor,context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding, context], dim=-1)  # [B,T,D]
        _, h = self.gru(x)
        return self.head(h.squeeze(0))