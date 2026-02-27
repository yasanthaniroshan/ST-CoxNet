import torch
from typing import Iterable, List, Tuple


class EpochRunner:
    """
    Common base class that implements the core run_epoch logic
    for both training and validation-style runners.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        loss: torch.nn.Module,
        loss_weights: Iterable[float],
    ):
        self.model = model
        self.device = device
        self.loss = loss
        self.loss_weights = list(loss_weights)

    def _compute_multihead_loss(
        self,
        rr_windows: torch.Tensor,
        hrv_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass + multi-head loss aggregation.
        """
        losses: List[torch.Tensor] = []

        c_seq = self.model(rr_windows)  # [B, T, context_dim]
        last_context = c_seq[:, -1, :]  # [B, context_dim]
        total_loss = torch.tensor(0.0, device=rr_windows.device)
        y_preds = self.model.predictor(last_context)

        for idx, (y_pred, weight) in enumerate(zip(y_preds, self.loss_weights)):
            loss = self.loss(y_pred, hrv_targets[:, idx, :])
            losses.append(loss)
            total_loss = total_loss + weight * loss
        
        return total_loss, tuple(losses)
    
    def _predict_multihead(self, rr_windows: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass + multi-head prediction (no loss).
        """
        c_seq = self.model(rr_windows)  # [B, T, context_dim]
        last_context = c_seq[:, -1, :]  # [B, context_dim]
        y_preds = self.model.predictor(last_context)
        return tuple(y_preds)

    def _run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
    ) -> Tuple[float, Tuple[float, ...]]:
        """
        Shared epoch loop used by Trainer and Validator.
        """
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        running_total = 0.0
        running_heads = [0.0 for _ in self.loss_weights]
        num_batches = 0

        with torch.set_grad_enabled(is_train):
            for rr_windows, hrv_targets, _ in dataloader:
                rr_windows = rr_windows.to(self.device)
                hrv_targets = hrv_targets.to(self.device)

                if is_train and optimizer is not None:
                    optimizer.zero_grad()

                total_loss, losses = self._compute_multihead_loss(
                    rr_windows=rr_windows,
                    hrv_targets=hrv_targets,
                )
                # print(f"Batch Loss: {total_loss.item():.4f} | " + " | ".join(
                #     [f"Head {idx+1}: {l.item():.4f}" for idx, l in enumerate(losses)]
                # ))

                if is_train and optimizer is not None:
                    total_loss.backward()
                    optimizer.step()

                running_total += float(total_loss.item())
                for idx, l in enumerate(losses):
                    running_heads[idx] += float(l.item())
                num_batches += 1

        avg_total = running_total / num_batches if num_batches > 0 else 0.0
        avg_heads = tuple(h / num_batches for h in running_heads) if num_batches > 0 else tuple(
            running_heads
        )
        return avg_total, avg_heads

