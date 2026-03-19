import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import wandb

from Utils.Dataset.CPCCoxDataset import CPCCoxDataset
from Model.SimpleTime import SimpleTimeEncoder, TimeToEventHead
from Loss.TimeContrastive import time_contrastive_loss


load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)


def time_regression_loss(pred_time: torch.Tensor, true_time: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_time, true_time)


def evaluate(
    encoder,
    head,
    data_loader,
    device,
    config,
    use_contrastive,
):
    encoder.eval()
    head.eval()

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_regression_loss = 0.0
    pred_all = []
    true_all = []

    with torch.no_grad():
        for rr, _, time, event in data_loader:
            rr = rr.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)

            latent, proj = encoder(rr)
            pred_time = head(latent)

            reg = time_regression_loss(pred_time, time)

            if use_contrastive:
                contrastive = time_contrastive_loss(
                    embeddings=proj,
                    times=time,
                    temperature=config["temperature"],
                    tau=config["time_tau"],
                    use_knn_positives=config["use_knn_positives"],
                    knn_k=config["knn_k"],
                )
                loss = config["contrastive_weight"] * contrastive + config["regression_weight"] * reg
                total_contrastive_loss += contrastive.item()
            else:
                loss = config["regression_weight"] * reg

            total_loss += loss.item()
            total_regression_loss += reg.item()

            pred_all.append(pred_time.cpu().numpy())
            true_all.append(time.cpu().numpy())

    pred_all = np.concatenate(pred_all, axis=0) if len(pred_all) > 0 else np.array([])
    true_all = np.concatenate(true_all, axis=0) if len(true_all) > 0 else np.array([])

    mae = float(np.mean(np.abs(pred_all - true_all))) if len(pred_all) > 0 else 0.0
    rmse = float(np.sqrt(np.mean((pred_all - true_all) ** 2))) if len(pred_all) > 0 else 0.0

    return {
        "loss": total_loss / max(len(data_loader), 1),
        "contrastive_loss": total_contrastive_loss / max(len(data_loader), 1),
        "regression_loss": total_regression_loss / max(len(data_loader), 1),
        "mae": mae,
        "rmse": rmse,
        "pred": pred_all,
        "true": true_all,
    }


def plot_training_curves(history, save_path):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history["epoch"], history["train_loss"], label="Train")
    ax1.plot(history["epoch"], history["val_loss"], label="Validation")
    ax1.set_title("Combined Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history["epoch"], history["train_contrastive"], label="Train")
    ax2.plot(history["epoch"], history["val_contrastive"], label="Validation")
    ax2.set_title("Time Contrastive Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history["epoch"], history["train_regression"], label="Train")
    ax3.plot(history["epoch"], history["val_regression"], label="Validation")
    ax3.set_title("Time Regression Loss (MSE)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history["epoch"], history["train_mae"], label="Train MAE")
    ax4.plot(history["epoch"], history["val_mae"], label="Validation MAE")
    ax4.set_title("Time Prediction MAE")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("MAE")
    ax4.grid(alpha=0.3)
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history["epoch"], history["val_rmse"], label="Validation RMSE")
    ax5.set_title("Validation RMSE")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("RMSE")
    ax5.grid(alpha=0.3)
    ax5.legend()

    warmup = history.get("contrastive_warmup_epochs", 0)
    if warmup > 0:
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.axvline(x=warmup, color="gray", linestyle="--", alpha=0.5, label="Contrastive on")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    return fig


def plot_scatter(true_time, pred_time, epoch):
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(true_time, pred_time, alpha=0.4, s=12)
    max_v = max(float(np.max(true_time)), float(np.max(pred_time))) if len(true_time) > 0 else 1.0
    plt.plot([0, max_v], [0, max_v], "r--")
    plt.title(f"Actual vs Predicted Time (epoch {epoch})")
    plt.xlabel("Actual time-to-event (normalized)")
    plt.ylabel("Predicted time-to-event")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


try:
    config = {
        "afib_length": 60 * 60,
        "sr_length": int(1.5 * 60 * 60),
        "number_of_windows_in_segment": 10,
        "stride": 20,
        "window_size": 100,
        "validation_split": 0.15,
        "batch_size": 256,
        "epochs": 50,
        "latent_dim": 128,
        "proj_dim": 64,
        "dropout": 0.1,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "grad_clip": 1.0,
        "temperature": 0.1,
        "time_tau": 0.05,
        "use_knn_positives": True,
        "knn_k": 4,
        "contrastive_weight": 0.005,
        "regression_weight": 1.0,
        "contrastive_warmup_epochs": 10,
    }

    run = wandb.init(
        entity="eml-labs",
        project="Simple-Time-Contrastive",
        config=config,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("plots", exist_ok=True)

    processed_dataset_path = "/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"

    train_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"],
        sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"],
        window_size=config["window_size"],
        validation_split=config["validation_split"],
        train=True,
    )
    validation_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"],
        sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"],
        window_size=config["window_size"],
        validation_split=config["validation_split"],
        train=False,
    )
    logger.info(f"Loaded {len(train_dataset)} training segments.")
    logger.info(f"Loaded {len(validation_dataset)} validation segments.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SimpleTimeEncoder(
        window_size=config["window_size"],
        latent_dim=config["latent_dim"],
        proj_dim=config["proj_dim"],
        dropout=config["dropout"],
    ).to(device)
    time_head = TimeToEventHead(
        latent_dim=config["latent_dim"],
        hidden_dim=128,
        dropout=config["dropout"],
    ).to(device)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(time_head.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["lr"] * 0.01,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_contrastive": [],
        "val_contrastive": [],
        "train_regression": [],
        "val_regression": [],
        "train_mae": [],
        "val_mae": [],
        "val_rmse": [],
        "contrastive_warmup_epochs": config["contrastive_warmup_epochs"],
    }

    best_val_mae = float("inf")
    warmup_epochs = config["contrastive_warmup_epochs"]

    pbar = tqdm(total=config["epochs"], desc="Training Simple Time Model")
    for epoch in range(config["epochs"]):
        encoder.train()
        time_head.train()

        use_contrastive = epoch >= warmup_epochs

        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_regression_loss = 0.0
        pred_all = []
        true_all = []

        for rr, _, time, event in train_loader:
            rr = rr.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)

            optimizer.zero_grad()
            latent, proj = encoder(rr)
            pred_time = time_head(latent)

            reg = time_regression_loss(pred_time, time)

            if use_contrastive:
                contrastive = time_contrastive_loss(
                    embeddings=proj,
                    times=time,
                    temperature=config["temperature"],
                    tau=config["time_tau"],
                    use_knn_positives=config["use_knn_positives"],
                    knn_k=config["knn_k"],
                )
                loss = config["contrastive_weight"] * contrastive + config["regression_weight"] * reg
                total_contrastive_loss += contrastive.item()
            else:
                loss = config["regression_weight"] * reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(time_head.parameters()),
                config["grad_clip"],
            )
            optimizer.step()

            total_loss += loss.item()
            total_regression_loss += reg.item()
            pred_all.append(pred_time.detach().cpu().numpy())
            true_all.append(time.detach().cpu().numpy())

        scheduler.step()

        pred_all = np.concatenate(pred_all, axis=0)
        true_all = np.concatenate(true_all, axis=0)
        train_mae = float(np.mean(np.abs(pred_all - true_all)))

        train_metrics = {
            "loss": total_loss / max(len(train_loader), 1),
            "contrastive_loss": total_contrastive_loss / max(len(train_loader), 1),
            "regression_loss": total_regression_loss / max(len(train_loader), 1),
            "mae": train_mae,
        }
        val_metrics = evaluate(
            encoder=encoder,
            head=time_head,
            data_loader=validation_loader,
            device=device,
            config=config,
            use_contrastive=use_contrastive,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_contrastive"].append(train_metrics["contrastive_loss"])
        history["val_contrastive"].append(val_metrics["contrastive_loss"])
        history["train_regression"].append(train_metrics["regression_loss"])
        history["val_regression"].append(val_metrics["regression_loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])

        scatter_fig = plot_scatter(val_metrics["true"], val_metrics["pred"], epoch + 1)
        phase = "warmup (reg only)" if not use_contrastive else "reg + contrastive"
        run.log(
            {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                "phase": phase,
                "train_loss": train_metrics["loss"],
                "train_contrastive_loss": train_metrics["contrastive_loss"],
                "train_regression_loss": train_metrics["regression_loss"],
                "train_mae": train_metrics["mae"],
                "val_loss": val_metrics["loss"],
                "val_contrastive_loss": val_metrics["contrastive_loss"],
                "val_regression_loss": val_metrics["regression_loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_scatter": wandb.Image(scatter_fig),
            }
        )
        plt.close(scatter_fig)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(encoder.state_dict(), "simple_time_encoder_best.pth")
            torch.save(time_head.state_dict(), "time_to_event_head_best.pth")

        pbar.update(1)
        pbar.set_description(f"Epoch {epoch + 1} [{phase}]")
        pbar.write(
            f"Epoch {epoch + 1} [{phase}] | "
            f"Train: loss={train_metrics['loss']:.4f} reg={train_metrics['regression_loss']:.4f} ctr={train_metrics['contrastive_loss']:.4f} | "
            f"Val: loss={val_metrics['loss']:.4f} reg={val_metrics['regression_loss']:.4f} MAE={val_metrics['mae']:.4f} RMSE={val_metrics['rmse']:.4f}"
        )

    pbar.close()

    curve_fig = plot_training_curves(history, save_path="plots/simple_time_training_curves.png")
    run.log({"training_curves": wandb.Image(curve_fig)})
    plt.close(curve_fig)

    encoder_artifact = wandb.Artifact("simple_time_encoder_best", type="model")
    encoder_artifact.add_file("simple_time_encoder_best.pth")
    run.log_artifact(encoder_artifact)

    head_artifact = wandb.Artifact("time_to_event_head_best", type="model")
    head_artifact.add_file("time_to_event_head_best.pth")
    run.log_artifact(head_artifact)

    logger.info(f"Training complete. Best Val MAE: {best_val_mae:.4f}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    if "run" in locals():
        run.finish()
