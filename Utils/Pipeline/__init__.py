from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from Device import Device
from Metadata import SplitMetadata, FileLoaderMetadata
from Train import Trainer
from Utils.Dataset.RRSequenceDataset import RRSequenceDataset
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


def _build_datasets_and_loaders(
    cfg: DictConfig,
    logger: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], FileLoaderMetadata]:
    """
    Shared construction of datasets and dataloaders from Hydra cfg.
    Returns train_loader, val_loader, test_loader, and the base file_loader metadata.
    """
    file_loader: FileLoaderMetadata = instantiate(cfg.dataset.file_loader)
    split_metadata = SplitMetadata(**cfg.split)

    train_files, val_files, test_files = split(file_loader.file_names, split_metadata)

    if logger is not None:
        logger.info(
            f"Files are split into Train: {len(train_files)}, "
            f"Val: {len(val_files)}, Test: {len(test_files)} | "
            f"Total: {len(file_loader.file_names)}"
        )

    train_file_loader_metadata = file_loader.model_copy(update={"file_names": train_files})
    val_file_loader_metadata = file_loader.model_copy(update={"file_names": val_files})
    test_file_loader_metadata = file_loader.model_copy(update={"file_names": test_files})

    sampling_rate = cfg.dataset.sampling_rate
    rr_sequence_config = instantiate(cfg.preprocessing.rr_sequence)
    feature_extractors_config = instantiate(cfg.feature_extractors)

    train_dataset = RRSequenceDataset(
        sampling_rate,
        rr_sequence_config,
        train_file_loader_metadata,
        feature_extractors_config,
    )
    val_dataset = RRSequenceDataset(
        sampling_rate,
        rr_sequence_config,
        val_file_loader_metadata,
        feature_extractors_config,
    )
    test_dataset = RRSequenceDataset(
        sampling_rate,
        rr_sequence_config,
        test_file_loader_metadata,
        feature_extractors_config,
    )

    if logger is not None:
        logger.info(
            f"Datasets created with lengths - "
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

    train_loader = DataLoader(train_dataset, **cfg.trainer.loader)
    val_loader = DataLoader(val_dataset, **cfg.validator.loader)
    test_loader: Optional[DataLoader] = None
    if "tester" in cfg and "loader" in cfg.tester:
        test_loader = DataLoader(test_dataset, **cfg.tester.loader)

    return train_loader, val_loader, test_loader, file_loader


def build_pipeline(
    cfg: DictConfig,
    logger: Optional[Any] = None,
) -> Pipeline:
    """
    Construct the full training/validation pipeline from a Hydra config.
    This centralizes all wiring so entry points can remain thin.
    """
    train_loader, val_loader, test_loader, _ = _build_datasets_and_loaders(cfg, logger=logger)

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

    return Pipeline(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        trainer=trainer,
        validator=validator,
    )

