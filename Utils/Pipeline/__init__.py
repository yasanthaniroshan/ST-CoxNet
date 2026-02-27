from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader,Subset
import numpy as np
from sklearn.preprocessing import StandardScaler

from Device import Device
from Metadata import SplitMetadata, FileLoaderMetadata
from Train import Trainer
from Utils.Dataset.RRSequenceDataset import RRSequenceDataset
from Utils.Dataset.RRSequenceCSVData import RRSequenceCSVData
from Utils.Dataset.RRSequenceCSVDataset import RRSequenceCSVDataset
from Utils.Visualizer.RegresssionPlotter import RegressionPlotter
from Utils.Dataset.Splitter import split
from Validator import Validator


@dataclass
class Pipeline:
    model: torch.nn.Module
    device: torch.device
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    trainer: Trainer
    validator: Optional[Validator]
    plotter : Optional[RegressionPlotter] = None



def _build_datasets_and_loaders(
    cfg: DictConfig,
    logger: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], FileLoaderMetadata]:
    """
    Shared construction of datasets and dataloaders from Hydra cfg.
    Returns train_loader, val_loader, test_loader, and the base file_loader metadata.
    """
    csv_data: RRSequenceCSVData = instantiate(cfg.dataset)
    split_metadata = instantiate(cfg.split)
    train_records, val_records, test_records = split(csv_data.records, split_metadata=split_metadata)

    train_hrv = np.vstack([rec["hrv"] for rec in train_records.values()])

    scaler = StandardScaler()
    scaler.fit(train_hrv)

    for records in [train_records, val_records, test_records]:
        for key in records:
            records[key]["hrv"] = scaler.transform(records[key]["hrv"])

    if logger is not None:
        logger.info(
            f"Data is split into Train: {len(train_records)}, "
            f"Val: {len(val_records)}, Test: {len(test_records)} | "
            f"Total: {len(csv_data.records)}"
        )


    train_dataset = RRSequenceCSVDataset(
        records=train_records,
        horizons=csv_data.horizons,
        seq_len=csv_data.seq_len
    )
    val_dataset = RRSequenceCSVDataset(
        records=val_records,
        horizons=csv_data.horizons,
        seq_len=csv_data.seq_len
    )
    test_dataset = RRSequenceCSVDataset(
        records=test_records,
        horizons=csv_data.horizons,
        seq_len=csv_data.seq_len
    )

    if logger is not None:
        logger.info(
            f"Datasets created with lengths - "
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

    # train_dataset = Subset(train_dataset, range(min(100, len(train_dataset))))
    train_loader = DataLoader(train_dataset, **cfg.trainer.loader)
    val_loader = DataLoader(val_dataset, **cfg.validator.loader)
    test_loader: Optional[DataLoader] = None
    if "tester" in cfg and "loader" in cfg.tester:
        test_loader = DataLoader(test_dataset, **cfg.tester.loader)

    return train_loader, val_loader, test_loader


def build_pipeline(
    cfg: DictConfig,
    logger: Optional[Any] = None,
) -> Pipeline:
    """
    Construct the full training/validation pipeline from a Hydra config.
    This centralizes all wiring so entry points can remain thin.
    """
    train_loader, val_loader, test_loader = _build_datasets_and_loaders(cfg, logger=logger)

    model: torch.nn.Module = instantiate(cfg.first_stage_model)
    if logger is not None:
        logger.info(f"Model instantiated: {model.__class__.__name__}")

    device_wrapper: Device = instantiate(cfg.device)
    device = device_wrapper.device

    optimizer: torch.optim.Optimizer = instantiate(cfg.trainer.optimizer, params=model.parameters())
    loss_fn: torch.nn.Module = instantiate(cfg.trainer.loss)

    trainer = Trainer(
        model,
        optimizer,
        device,
        loss_fn,
        number_of_predictors=cfg.first_stage_model.config.predictor.num_heads,
        loss_weights=cfg.trainer.loss_weights,
    )

    validator: Optional[Validator] = None
    if hasattr(cfg, "validator"):
        validator = Validator(
            model,
            device,
            loss_fn,
            number_of_predictors=cfg.first_stage_model.config.predictor.num_heads,
        )

    model.to(device)

    plotter = RegressionPlotter()

    return Pipeline(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        trainer=trainer,
        validator=validator,
        plotter=plotter
    )

