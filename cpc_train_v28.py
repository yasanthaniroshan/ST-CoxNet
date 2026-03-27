"""
v28 — Multi-patient TTE regression on excellent patients only.

Same architecture as v16 (Encoder → GRU → MLP → TTE in minutes),
but training and validation are restricted to patients whose
per-patient v27 models were classified as "excellent".
"""

import json
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


def compute_per_patient_rho(preds, actual, pids, min_segments=10):
    """Spearman ρ per patient, returns dict of pid → ρ."""
    results = {}
    for pid in np.unique(pids):
        m = pids == pid
        if m.sum() < min_segments:
            continue
        p, a = preds[m], actual[m]
        if a.max() - a.min() < 1e-6:
            continue
        rho, _ = spearmanr(p, a)
        if not np.isnan(rho):
            results[int(pid)] = rho
    return results


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
        "batch_size": 64,
        "lr": 1e-3,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "patience": 20,
    }

    run = wandb.init(
        entity="eml-labs", project="CPC-New-Temporal-Ranking",
        name="v17-excellent-patients-only", config=config,
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Load excellent patient IDs from v15c JSON ──
    with open("v15c_patient_quality.json", "r") as f:
        quality = json.load(f)
    excellent_pids = {int(p["pid"]) for p in quality["patients"] if p["tier"] == "excellent"}

    logger.info(
        f"Using excellent patients only: {len(excellent_pids)} / "
        f"{quality['summary']['n_patients']} "
        f"(ρ≥0.9 in per-patient v15c models)"
    )

    # ══════════════════════════════════════════════════════════════════
    #  Load data — SR-only, then filter to excellent patients
    # ══════════════════════════════════════════════════════════════════
    processed = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    ds_args = dict(
        processed_dataset_path=processed,
        afib_length=config["afib_length"],
        sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"],
        window_size=config["window_size"],
        validation_split=config["validation_split"],
    )

    train_ds_full = CPCTemporalDataset(**ds_args, train=True, sr_only=True)
    val_ds_full = CPCTemporalDataset(**ds_args, train=False, sr_only=True)

    # Combine train+val SR segments, then restrict to excellent patients,
    # THEN do a random segment-level split (data leakage by design).
    all_data = torch.cat([train_ds_full.data, val_ds_full.data], dim=0)
    all_times = torch.cat([train_ds_full.times, val_ds_full.times], dim=0)
    all_pids = torch.cat([train_ds_full.patient_ids, val_ds_full.patient_ids], dim=0)

    all_pids_np = all_pids.numpy()
    mask_excellent = np.isin(all_pids_np, list(excellent_pids))

    all_data = all_data[mask_excellent]
    all_tte_min = all_times[mask_excellent].float().clamp(min=0) / 60.0
    all_pids = all_pids[mask_excellent]

    N = len(all_data)
    idx = torch.randperm(N)
    split = int(N * (1 - config["validation_split"]))

    train_idx = idx[:split]
    val_idx = idx[split:]

    train_data = all_data[train_idx]
    train_tte_min = all_tte_min[train_idx]
    train_pids = all_pids[train_idx]

    val_data = all_data[val_idx]
    val_tte_min = all_tte_min[val_idx]
    val_pids = all_pids[val_idx]

    n_train_patients = len(np.unique(train_pids.numpy()))
    n_val_patients = len(np.unique(val_pids.numpy()))
    max_tte_min = float(all_tte_min.max()) if len(all_tte_min) > 0 else 0.0

    logger.info(
        f"Excellent-only SEGMENT split (leaky): {n_train_patients} train patients, "
        f"{n_val_patients} val patients (segments from same patients may be in both)."
    )
    logger.info(
        f"Train (excellent only): {len(train_data)} SR segments | "
        f"Val (excellent only): {len(val_data)} SR segments"
    )
    logger.info(f"TTE range (excellent only): 0 — {max_tte_min:.1f} min")

    train_loader = DataLoader(
        TensorDataset(train_data, train_tte_min, train_pids),
        batch_size=config["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_tte_min, val_pids),
        batch_size=config["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TTEPredictor(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,} | Device: {device}")

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01,
    )
    loss_fn = nn.SmoothL1Loss()

    # ══════════════════════════════════════════════════════════════════
    #  Training loop
    # ══════════════════════════════════════════════════════════════════
    pbar = tqdm(total=config["epochs"], desc="Training (excellent only)")
    best_val_mae = float("inf")
    patience_ctr = 0

    for epoch in range(config["epochs"]):
        model.train()
        t_loss, t_n = 0.0, 0
        t_preds, t_actual, t_pids_ep = [], [], []

        for rr, tte, pid in train_loader:
            rr, tte = rr.to(device), tte.to(device)
            optimizer.zero_grad()
            pred = model(rr)
            loss = loss_fn(pred, tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item() * rr.size(0)
            t_n += rr.size(0)
            t_preds.append(pred.detach().cpu().numpy())
            t_actual.append(tte.cpu().numpy())
            t_pids_ep.append(pid.numpy())

        scheduler.step()

        t_preds = np.concatenate(t_preds)
        t_actual = np.concatenate(t_actual)
        t_pids_ep = np.concatenate(t_pids_ep)
        t_mae = np.mean(np.abs(t_preds - t_actual))
        t_rho, _ = spearmanr(t_preds, t_actual)
        t_rho = t_rho if not np.isnan(t_rho) else 0.0

        t_per_patient = compute_per_patient_rho(t_preds, t_actual, t_pids_ep)
        t_mean_pp_rho = float(np.mean(list(t_per_patient.values()))) if t_per_patient else 0.0

        # ── Validation ──
        model.eval()
        v_loss, v_n = 0.0, 0
        v_preds, v_actual, v_pids_ep = [], [], []

        with torch.no_grad():
            for rr, tte, pid in val_loader:
                rr, tte = rr.to(device), tte.to(device)
                pred = model(rr)
                v_loss += loss_fn(pred, tte).item() * rr.size(0)
                v_n += rr.size(0)
                v_preds.append(pred.cpu().numpy())
                v_actual.append(tte.cpu().numpy())
                v_pids_ep.append(pid.numpy())

        v_preds = np.concatenate(v_preds)
        v_actual = np.concatenate(v_actual)
        v_pids_ep = np.concatenate(v_pids_ep)
        v_mae = np.mean(np.abs(v_preds - v_actual))
        v_rho, _ = spearmanr(v_preds, v_actual)
        v_rho = v_rho if not np.isnan(v_rho) else 0.0

        ss_res = np.sum((v_actual - v_preds) ** 2)
        ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
        v_r2 = 1 - ss_res / max(ss_tot, 1e-8)

        v_per_patient = compute_per_patient_rho(v_preds, v_actual, v_pids_ep)
        v_mean_pp_rho = float(np.mean(list(v_per_patient.values()))) if v_per_patient else 0.0

        # ── Early stopping ──
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            patience_ctr = 0
            torch.save(model.state_dict(), "v17_best.pth")
        else:
            patience_ctr += 1

        run.log({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train/loss": t_loss / t_n,
            "train/mae_min": t_mae,
            "train/spearman_global": t_rho,
            "train/spearman_per_patient": t_mean_pp_rho,
            "val/loss": v_loss / v_n,
            "val/mae_min": v_mae,
            "val/spearman_global": v_rho,
            "val/spearman_per_patient": v_mean_pp_rho,
            "val/r2": v_r2,
        })

        if (epoch + 1) % 10 == 0 or epoch == config["epochs"] - 1:
            pbar.write(
                f"Epoch {epoch+1:>3} | "
                f"MAE {t_mae:.2f}/{v_mae:.2f} | "
                f"ρ_global {t_rho:.3f}/{v_rho:.3f} | "
                f"ρ_patient {t_mean_pp_rho:.3f}/{v_mean_pp_rho:.3f} | "
                f"R² {v_r2:.3f}"
            )
        pbar.update(1)

        if patience_ctr >= config["patience"]:
            pbar.write(
                f"Early stopping at epoch {epoch+1} "
                f"(best val MAE: {best_val_mae:.2f} min)"
            )
            break

    pbar.close()

    # ══════════════════════════════════════════════════════════════════
    #  Final evaluation (best model on validation set)
    # ══════════════════════════════════════════════════════════════════
    model.load_state_dict(torch.load("v17_best.pth", weights_only=True))
    model.eval()

    v_preds, v_actual, v_pids_all = [], [], []
    with torch.no_grad():
        for rr, tte, pid in val_loader:
            pred = model(rr.to(device))
            v_preds.append(pred.cpu().numpy())
            v_actual.append(tte.numpy())
            v_pids_all.append(pid.numpy())

    v_preds = np.concatenate(v_preds)
    v_actual = np.concatenate(v_actual)
    v_pids_all = np.concatenate(v_pids_all)

    global_mae = np.mean(np.abs(v_preds - v_actual))
    global_rho, _ = spearmanr(v_preds, v_actual)
    ss_res = np.sum((v_actual - v_preds) ** 2)
    ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
    global_r2 = 1 - ss_res / max(ss_tot, 1e-8)

    pp_rhos = compute_per_patient_rho(v_preds, v_actual, v_pids_all)
    mean_pp_rho = float(np.mean(list(pp_rhos.values()))) if pp_rhos else 0.0

    logger.info(f"\n{'='*65}")
    logger.info("FINAL — best model on validation set (excellent patients only)")
    logger.info(f"  Global MAE:            {global_mae:.2f} min")
    logger.info(f"  Global Spearman ρ:     {global_rho:.4f}")
    logger.info(f"  Global R²:             {global_r2:.4f}")
    logger.info(f"  Per-patient ρ (mean):  {mean_pp_rho:.4f}  ({len(pp_rhos)} patients)")
    logger.info(f"  TTE range:             0 — {max_tte_min:.1f} min")
    logger.info(f"  Best val MAE:          {best_val_mae:.2f} min")
    logger.info(f"{'='*65}")

    # ══════════════════════════════════════════════════════════════════
    #  Diagnostic plots (same layout as v16)
    # ══════════════════════════════════════════════════════════════════
    unique_val_pids = np.unique(v_pids_all)
    pid_colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_val_pids), 1)))
    pid_cmap = {pid: pid_colors[i] for i, pid in enumerate(unique_val_pids)}

    fig = plt.figure(figsize=(28, 10))
    fig.suptitle(
        f"v17 Excellent-Only TTE  |  MAE={global_mae:.2f} min  |  "
        f"ρ_global={global_rho:.3f}  |  ρ_patient={mean_pp_rho:.3f}  |  "
        f"R²={global_r2:.3f}  |  {len(pp_rhos)} val patients",
        fontsize=14,
    )

    # 1 — Actual vs Predicted (colored by patient)
    ax = fig.add_subplot(1, 5, 1)
    for pid in unique_val_pids:
        m = v_pids_all == pid
        ax.scatter(v_actual[m], v_preds[m], alpha=0.5, s=12, c=[pid_cmap[pid]], label=f"P{pid}")
    lim = max(v_actual.max(), v_preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual TTE (min)")
    ax.set_ylabel("Predicted TTE (min)")
    ax.set_title("Actual vs Predicted")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # 2 — Residuals vs Actual
    ax = fig.add_subplot(1, 5, 2)
    residuals = v_preds - v_actual
    for pid in unique_val_pids:
        m = v_pids_all == pid
        ax.scatter(v_actual[m], residuals[m], alpha=0.5, s=12, c=[pid_cmap[pid]])
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Actual TTE (min)")
    ax.set_ylabel("Error (min)")
    ax.set_title(f"Residuals (mean={residuals.mean():.2f})")
    ax.grid(alpha=0.3)

    # 3 — Per-patient Spearman ρ bar chart
    ax = fig.add_subplot(1, 5, 3)
    if pp_rhos:
        sorted_pids = sorted(pp_rhos.keys(), key=lambda p: pp_rhos[p])
        rho_vals = [pp_rhos[p] for p in sorted_pids]
        bar_colors = ["steelblue" if r >= 0 else "coral" for r in rho_vals]
        ax.barh(range(len(sorted_pids)), rho_vals, color=bar_colors, height=0.7)
        ax.set_yticks(range(len(sorted_pids)))
        ax.set_yticklabels([f"P{p}" for p in sorted_pids], fontsize=7)
        ax.axvline(mean_pp_rho, color="k", ls="--", lw=1, label=f"mean={mean_pp_rho:.3f}")
        ax.set_xlabel("Spearman ρ")
        ax.set_title("Per-Patient ρ")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4 — TTE trajectory (sorted by actual)
    ax = fig.add_subplot(1, 5, 4)
    order = np.argsort(-v_actual)
    ax.plot(v_actual[order], label="Actual", alpha=0.8, lw=1.0)
    ax.plot(v_preds[order], label="Predicted", alpha=0.8, lw=1.0)
    ax.set_xlabel("Segment (far → close to AF)")
    ax.set_ylabel("TTE (min)")
    ax.set_title("TTE trajectory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 5 — Error distribution
    ax = fig.add_subplot(1, 5, 5)
    ax.hist(residuals, bins=60, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(residuals.mean(), color="red", ls="--", lw=1, label=f"mean={residuals.mean():.2f}")
    med = np.median(residuals)
    ax.axvline(med, color="orange", ls="--", lw=1, label=f"median={med:.2f}")
    ax.set_xlabel("Prediction Error (min)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("v17_excellent_only_results.png", dpi=150, bbox_inches="tight")
    run.log({
        "final/global_mae": global_mae,
        "final/global_spearman": global_rho,
        "final/global_r2": global_r2,
        "final/per_patient_spearman": mean_pp_rho,
        "final/n_val_patients": len(pp_rhos),
        "final_plot": wandb.Image(fig),
    })
    plt.close(fig)

    logger.info("Plot saved to v17_excellent_only_results.png")

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

