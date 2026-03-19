"""
v15 — Single-patient TTE prediction sanity check.

Goal: Validate that RR intervals contain learnable signal for time-to-AF-onset.
  - One patient only (the one with the most segments)
  - One end-to-end model: Encoder → GRU → MLP → TTE (minutes)
  - Direct regression with Huber loss, no contrastive/Cox machinery
  - If this works → the signal exists, multi-patient just needs generalisation
  - If this fails → deeper problem with data or architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import wandb
import logging
from tqdm import tqdm

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TTEPredictor(nn.Module):
    """Encoder → GRU → MLP → scalar TTE prediction (minutes)."""

    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, rr_windows):
        B, T, W = rr_windows.shape
        z = self.encoder(rr_windows.view(B * T, W)).view(B, T, -1)
        _, h = self.gru(z)
        return self.head(h.squeeze(0)).squeeze(-1)  # [B]


try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    config = {
        "afib_length": 60 * 60,
        "sr_length": int(1.5 * 60 * 60),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        # Training
        "epochs": 200,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "val_split": 0.2,
    }

    run = wandb.init(
        entity="eml-labs", project="CPC-New-Temporal-Ranking",
        name="v15-single-patient-tte", config=config,
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # ══════════════════════════════════════════════════════════════════
    #  Load data & select single patient
    # ══════════════════════════════════════════════════════════════════
    processed = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    ds = CPCTemporalDataset(
        processed_dataset_path=processed,
        afib_length=config["afib_length"],
        sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"],
        window_size=config["window_size"],
        validation_split=config["validation_split"],
        train=True,
    )

    pids_np = ds.patient_ids.numpy()
    unique_pids, counts = np.unique(pids_np, return_counts=True)

    logger.info(f"Available patients: {len(unique_pids)}")
    top5 = np.argsort(-counts)[:5]
    for i in top5:
        pid = unique_pids[i]
        n = counts[i]
        n_sr = int((ds.labels[pids_np == pid] == -1).sum())
        logger.info(f"  Patient {pid}: {n} segments ({n_sr} SR)")

    target_pid = unique_pids[np.argmax(counts)]
    mask = pids_np == target_pid

    data = ds.data[mask]
    labels = ds.labels[mask]
    times_sec = ds.times[mask].float().clamp(min=0)
    tte_min = times_sec / 60.0

    n_sr = int((labels == -1).sum())
    n_mix = int((labels == 0).sum())
    n_af = int((labels == 1).sum())
    max_tte = float(tte_min.max())

    logger.info(
        f"\nSelected patient {target_pid}: {len(data)} segments "
        f"(SR={n_sr}, Mixed={n_mix}, AF={n_af})"
    )
    logger.info(f"TTE range: 0 — {max_tte:.1f} min")

    # ── Train / Val split (random) ──
    N = len(data)
    idx = torch.randperm(N)
    split = int(N * (1 - config["val_split"]))

    train_data, val_data = data[idx[:split]], data[idx[split:]]
    train_tte, val_tte = tte_min[idx[:split]], tte_min[idx[split:]]
    train_lab, val_lab = labels[idx[:split]], labels[idx[split:]]

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_loader = DataLoader(
        TensorDataset(train_data, train_tte),
        batch_size=config["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_tte),
        batch_size=config["batch_size"], shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TTEPredictor(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01,
    )
    loss_fn = nn.SmoothL1Loss()

    # ══════════════════════════════════════════════════════════════════
    #  Training loop
    # ══════════════════════════════════════════════════════════════════
    pbar = tqdm(total=config["epochs"], desc="Training")
    best_val_mae = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        t_loss, t_preds, t_actual = 0.0, [], []

        for rr, tte in train_loader:
            rr, tte = rr.to(device), tte.to(device)
            optimizer.zero_grad()
            pred = model(rr)
            loss = loss_fn(pred, tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item() * rr.size(0)
            t_preds.append(pred.detach().cpu())
            t_actual.append(tte.cpu())

        scheduler.step()

        t_preds = torch.cat(t_preds).numpy()
        t_actual = torch.cat(t_actual).numpy()
        t_mae = np.mean(np.abs(t_preds - t_actual))
        t_rho, _ = spearmanr(t_preds, t_actual)
        t_rho = t_rho if not np.isnan(t_rho) else 0.0

        # ── Validation ──
        model.eval()
        v_loss, v_preds, v_actual = 0.0, [], []

        with torch.no_grad():
            for rr, tte in val_loader:
                rr, tte = rr.to(device), tte.to(device)
                pred = model(rr)
                v_loss += loss_fn(pred, tte).item() * rr.size(0)
                v_preds.append(pred.cpu())
                v_actual.append(tte.cpu())

        v_preds = torch.cat(v_preds).numpy()
        v_actual = torch.cat(v_actual).numpy()
        v_mae = np.mean(np.abs(v_preds - v_actual))
        v_rho, _ = spearmanr(v_preds, v_actual)
        v_rho = v_rho if not np.isnan(v_rho) else 0.0

        ss_res = np.sum((v_actual - v_preds) ** 2)
        ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
        v_r2 = 1 - ss_res / max(ss_tot, 1e-8)

        if v_mae < best_val_mae:
            best_val_mae = v_mae
            torch.save(model.state_dict(), "tte_best.pth")

        run.log({
            "epoch": epoch + 1,
            "train/loss": t_loss / len(train_data),
            "train/mae_min": t_mae,
            "train/spearman": t_rho,
            "val/loss": v_loss / len(val_data),
            "val/mae_min": v_mae,
            "val/spearman": v_rho,
            "val/r2": v_r2,
        })

        if (epoch + 1) % 10 == 0 or epoch == config["epochs"] - 1:
            pbar.write(
                f"Epoch {epoch+1:>3} | "
                f"MAE {t_mae:.2f}/{v_mae:.2f} min | "
                f"ρ {t_rho:.3f}/{v_rho:.3f} | "
                f"R² {v_r2:.3f}"
            )
        pbar.update(1)

    pbar.close()

    # ══════════════════════════════════════════════════════════════════
    #  Final evaluation on ALL segments (best model)
    # ══════════════════════════════════════════════════════════════════
    model.load_state_dict(torch.load("tte_best.pth", weights_only=True))
    model.eval()

    full_loader = DataLoader(
        TensorDataset(data, tte_min, labels.float()),
        batch_size=64, shuffle=False,
    )

    all_preds, all_actual, all_lab = [], [], []
    with torch.no_grad():
        for rr, tte, lab in full_loader:
            pred = model(rr.to(device))
            all_preds.append(pred.cpu().numpy())
            all_actual.append(tte.numpy())
            all_lab.append(lab.numpy().astype(int))

    all_preds = np.concatenate(all_preds)
    all_actual = np.concatenate(all_actual)
    all_lab = np.concatenate(all_lab)

    sr = all_lab == -1
    sr_mae = np.mean(np.abs(all_preds[sr] - all_actual[sr]))
    sr_rho, _ = spearmanr(all_preds[sr], all_actual[sr])
    ss_r = np.sum((all_actual[sr] - all_preds[sr]) ** 2)
    ss_t = np.sum((all_actual[sr] - all_actual[sr].mean()) ** 2)
    sr_r2 = 1 - ss_r / max(ss_t, 1e-8)

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL (best model, all data)")
    logger.info(f"  SR segments: MAE={sr_mae:.2f} min | ρ={sr_rho:.3f} | R²={sr_r2:.3f}")
    logger.info(f"  TTE range:   0 — {max_tte:.1f} min")
    logger.info(f"  Best val MAE: {best_val_mae:.2f} min")
    logger.info(f"{'='*60}")

    # ── Plots ──
    fig = plt.figure(figsize=(22, 5))
    fig.suptitle(
        f"Single-Patient TTE  |  MAE={sr_mae:.2f} min  |  "
        f"ρ={sr_rho:.3f}  |  R²={sr_r2:.3f}  |  "
        f"range 0–{max_tte:.0f} min",
        fontsize=13,
    )

    # 1. Actual vs Predicted (SR)
    ax = fig.add_subplot(1, 4, 1)
    ax.scatter(all_actual[sr], all_preds[sr], alpha=0.5, s=14, c="steelblue")
    lim = max(all_actual.max(), all_preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual TTE (min)"); ax.set_ylabel("Predicted TTE (min)")
    ax.set_title("SR: Actual vs Predicted"); ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # 2. Residuals
    ax = fig.add_subplot(1, 4, 2)
    res = all_preds[sr] - all_actual[sr]
    ax.scatter(all_actual[sr], res, alpha=0.5, s=14, c="coral")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Actual TTE (min)"); ax.set_ylabel("Error (min)")
    ax.set_title(f"Residuals (mean={res.mean():.2f})"); ax.grid(alpha=0.3)

    # 3. Temporal order (all segments sorted by actual TTE)
    ax = fig.add_subplot(1, 4, 3)
    order = np.argsort(-all_actual)
    ax.plot(all_actual[order], label="Actual", alpha=0.8, lw=1.2)
    ax.plot(all_preds[order], label="Predicted", alpha=0.8, lw=1.2)
    ax.set_xlabel("Segment (far → close to AF)"); ax.set_ylabel("TTE (min)")
    ax.set_title("TTE trajectory"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 4. By segment type
    ax = fig.add_subplot(1, 4, 4)
    for lbl, name, col in [(-1, "SR", "steelblue"), (0, "Mixed", "green"), (1, "AF", "red")]:
        m = all_lab == lbl
        if m.any():
            ax.scatter(all_actual[m], all_preds[m], alpha=0.5, s=14, c=col, label=name)
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(min(all_preds.min(), 0) - 1, lim)
    ax.set_xlabel("Actual TTE (min)"); ax.set_ylabel("Predicted TTE (min)")
    ax.set_title("All segment types"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("v15_single_patient_results.png", dpi=150, bbox_inches="tight")
    run.log({"final_plot": wandb.Image(fig)})
    plt.close(fig)

    logger.info("Plot saved to v15_single_patient_results.png")

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
    if "pbar" in locals():
        pbar.close()
finally:
    if "run" in locals():
        run.finish()
