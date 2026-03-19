import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalRankingLoss(nn.Module):
    """
    Ranking-based temporal contrastive loss for time-to-event aware representations.

    Supports two modes:
      1. **Within-patient** (P, K args provided): ranking computed independently
         inside each of the P patient groups of K segments, then averaged.
         This is the recommended mode — it forces the encoder to learn
         TTE-relevant features rather than patient identity.
      2. **Batch-level** (P, K omitted): ranking computed across the full batch.
         Useful as a fallback or for flat-batch evaluation.
    """

    def __init__(self, temperature: float = 0.07, sigma: float = 0.15):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma

    def forward(self, z, tte, P=None, K=None):
        if P is not None and K is not None:
            return self._within_patient(z, tte, P, K)
        return self._batch_level(z, tte)

    # ──────────────────────────────────────────────────────────────────
    #  Within-patient mode  (vectorized over P patients)
    # ──────────────────────────────────────────────────────────────────
    def _within_patient(self, z, tte, P, K):
        device = z.device

        z = F.normalize(z.view(P, K, -1), dim=-1)     # [P, K, D]
        tte = tte.view(P, K)                            # [P, K]

        # Per-patient TTE normalisation to [0, 1]
        tte_min = tte.min(dim=1, keepdim=True).values
        tte_max = tte.max(dim=1, keepdim=True).values
        tte_range = (tte_max - tte_min).clamp(min=1e-6)
        tte_norm = (tte - tte_min) / tte_range           # [P, K]

        # Pairwise within-patient cosine similarity / τ
        sim = torch.bmm(z, z.transpose(1, 2)) / self.temperature   # [P, K, K]

        # Pairwise normalised TTE distance
        tte_diff = torch.abs(
            tte_norm.unsqueeze(1) - tte_norm.unsqueeze(2)
        )                                                            # [P, K, K]

        # Gaussian soft-positive weights
        pos_w = torch.exp(-tte_diff.pow(2) / (2 * self.sigma ** 2))

        # Mask diagonal (self-similarity)
        diag = ~torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)  # [1,K,K]
        pos_w = pos_w * diag.float()
        pos_w = pos_w / pos_w.sum(dim=2, keepdim=True).clamp(min=1e-8)

        # Contrastive loss per patient
        sim_masked = sim.masked_fill(~diag, float("-inf"))
        log_prob = F.log_softmax(sim_masked, dim=2)
        log_prob = log_prob.masked_fill(~diag, 0.0)       # avoid 0 * -inf = NaN

        loss_per_patient = -(pos_w * log_prob).sum(dim=2).mean(dim=1)  # [P]

        # Only count patients whose TTE actually varies
        valid = (tte_max.squeeze(1) - tte_min.squeeze(1)) > 1e-6       # [P]

        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        loss = loss_per_patient[valid].mean()

        # Metrics
        with torch.no_grad():
            raw_sim = torch.bmm(z, z.transpose(1, 2))
            close = (tte_diff < 0.15) & diag
            far   = (tte_diff > 0.50) & diag
            mean_close = raw_sim[close].mean().item() if close.any() else 0.0
            mean_far   = raw_sim[far].mean().item()   if far.any()   else 0.0

        return loss, {
            "mean_close_sim": mean_close,
            "mean_far_sim":   mean_far,
            "sim_gap":        mean_close - mean_far,
            "valid_patients":  int(valid.sum().item()),
        }

    # ──────────────────────────────────────────────────────────────────
    #  Batch-level mode  (original flat-batch ranking)
    # ──────────────────────────────────────────────────────────────────
    def _batch_level(self, z, tte):
        B = z.size(0)
        device = z.device

        if B <= 2:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        z = F.normalize(z, dim=-1)

        tte_min = tte.min()
        tte_range = (tte.max() - tte_min).clamp(min=1e-6)
        tte_norm = (tte - tte_min) / tte_range

        sim = torch.mm(z, z.t()) / self.temperature
        tte_diff = torch.abs(tte_norm.unsqueeze(0) - tte_norm.unsqueeze(1))

        pos_w = torch.exp(-tte_diff.pow(2) / (2 * self.sigma ** 2))

        diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_w = pos_w * diag_mask.float()
        pos_w = pos_w / pos_w.sum(dim=1, keepdim=True).clamp(min=1e-8)

        sim_masked = sim.masked_fill(~diag_mask, float("-inf"))
        log_prob = F.log_softmax(sim_masked, dim=1)
        log_prob = log_prob.masked_fill(~diag_mask, 0.0)

        loss = -(pos_w * log_prob).sum(dim=1).mean()

        with torch.no_grad():
            raw_sim = torch.mm(z, z.t())
            close = (tte_diff < 0.15) & diag_mask
            far   = (tte_diff > 0.50) & diag_mask
            mean_close = raw_sim[close].mean().item() if close.any() else 0.0
            mean_far   = raw_sim[far].mean().item()   if far.any()   else 0.0

        return loss, {
            "mean_close_sim": mean_close,
            "mean_far_sim":   mean_far,
            "sim_gap":        mean_close - mean_far,
        }
