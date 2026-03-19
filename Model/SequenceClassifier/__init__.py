import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Single-head attention that learns which positions in the sequence
    are most informative for classifying the last segment."""

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_seq, last_hidden):
        """
        h_seq: (B, K, D) -- full sequence hidden states
        last_hidden: (B, D) -- query from last position
        Returns: (B, D) -- attended context
        """
        q = self.query(last_hidden).unsqueeze(1)
        k = self.key(h_seq)
        attn = (q * k).sum(dim=-1) / self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        return (attn.unsqueeze(-1) * h_seq).sum(dim=1)


class SequenceTimeBinClassifier(nn.Module):
    """Lightweight GRU + attention classifier over a sequence of segment embeddings.

    Much smaller than a Transformer to prevent overfitting on the small
    effective dataset (~107 patients, ~6k sequences).
    """

    def __init__(
        self,
        per_segment_dim: int,
        num_classes: int,
        seq_len: int = 16,
        d_model: int = 64,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Sequential(
            nn.Linear(per_segment_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.gru_proj = nn.Linear(d_model * 2, d_model)
        self.gru_norm = nn.LayerNorm(d_model)
        self.gru_drop = nn.Dropout(dropout)

        self.attention = TemporalAttention(d_model, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, segment_features):
        """
        Args:
            segment_features: (B, K, per_segment_dim)
        Returns:
            logits: (B, num_classes)
        """
        x = self.input_proj(segment_features)

        h_seq, _ = self.gru(x)
        h_seq = self.gru_drop(self.gru_norm(self.gru_proj(h_seq)))

        last_h = h_seq[:, -1, :]
        attn_ctx = self.attention(h_seq, last_h)

        combined = torch.cat([last_h, attn_ctx], dim=-1)
        return self.classifier(combined)
