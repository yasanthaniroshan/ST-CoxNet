import torch
from typing import List, Tuple

from Utils.Runner.EpochRunner import EpochRunner


class Validator(EpochRunner):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        loss: torch.nn.Module,
        number_of_predictors: int,
        loss_weights: List[float] | None = None,
    ):
        weights = loss_weights if loss_weights is not None else [
            1.0 for _ in range(number_of_predictors)
        ]
        super().__init__(model=model, device=device, loss=loss, loss_weights=weights)

    def validation_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[float, Tuple[float, ...]]:
        return self._run_epoch(
            dataloader=dataloader,
            is_train=False,
            optimizer=None,
        )

