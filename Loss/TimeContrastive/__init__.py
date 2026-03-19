import torch
import torch.nn.functional as F


def time_contrastive_loss(
    embeddings: torch.Tensor,
    times: torch.Tensor,
    temperature: float = 0.1,
    tau: float = 0.1,
    use_knn_positives: bool = True,
    knn_k: int = 4,
) -> torch.Tensor:
    """
    Soft time-aware contrastive loss.
    embeddings: [B, D] (L2-normalized or not)
    times:      [B] normalized time-to-event
    """
    if embeddings.size(0) != times.size(0):
        raise ValueError(
            f"Batch mismatch: embeddings={embeddings.size(0)} times={times.size(0)}"
        )
    if embeddings.size(0) < 2:
        return embeddings.new_tensor(0.0)

    z = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(z, z.t()) / temperature

    time_diff = torch.abs(times[:, None] - times[None, :])
    weights = torch.exp(-time_diff / tau)

    diag_mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag_mask, -1e9)
    weights = weights.masked_fill(diag_mask, 0.0)

    if use_knn_positives:
        k = min(int(knn_k), z.size(0) - 1)
        if k > 0:
            knn_idx = torch.topk(
                time_diff.masked_fill(diag_mask, float("inf")),
                k=k,
                dim=1,
                largest=False,
            ).indices
            knn_mask = torch.zeros_like(weights, dtype=torch.bool)
            knn_mask.scatter_(1, knn_idx, True)
            weights = torch.where(knn_mask, weights, torch.zeros_like(weights))

    log_prob = F.log_softmax(sim, dim=1)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    loss = -(weights * log_prob).sum(dim=1)
    return loss.mean()
