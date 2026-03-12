import torch
import torch.nn.functional as F
from typing import List,Dict,Tuple
from Metadata import TrainerConfig


class Trainer:
    loss_weights = []
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,loss:torch.nn.Module,number_of_predictors:int,loss_weights:List[float] = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_weights = loss_weights if loss_weights is not None else [1.0 for _ in range(number_of_predictors)]
        if len(self.loss_weights) != number_of_predictors:
            raise ValueError(f"Length of loss_weights must match number_of_predictors. Got {len(self.loss_weights)} and {number_of_predictors}")
        self.loss = loss

    def training_step(
        self, 
        rr_windows: torch.Tensor, 
        hrv_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        rr_windows: [B, T, W]  -> RR windows
        hrv_targets : [B, T, num_metrics] -> HRV targets for different horizons
        Returns:
            total_loss: scalar tensor
            losses: tuple of individual losses
        """
        losses = []
        # Get context embeddings from model
        c_seq = self.model(rr_windows)  # [B, T, context_dim]
        last_context = c_seq[:, -1, :]  # [B, context_dim]
        total_loss = 0.0

        y_preds = self.model.predictor(last_context)  # Each: [B, num_metrics]
        for idx,y_pred in enumerate(y_preds):
            loss = self.loss(y_pred, hrv_targets[:, idx, :])
            losses.append(loss)
            total_loss += self.loss_weights[idx] * loss
    
        return total_loss,tuple(losses)
    

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        lossses = [0.0 for _ in self.loss_weights]
        num_batches = 0

        for rr_windows, hrv_targets, _ in dataloader:
            rr_windows = rr_windows.to(self.device)  # [B, T, W]
            hrv_targets = hrv_targets.to(self.device)  # [B, T, num_metrics]

            self.optimizer.zero_grad()
            loss, losses = self.training_step(rr_windows, hrv_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for idx, l in enumerate(losses):
                lossses[idx] += l.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_losses = [l / num_batches for l in lossses]

        return avg_loss, tuple(avg_losses)
