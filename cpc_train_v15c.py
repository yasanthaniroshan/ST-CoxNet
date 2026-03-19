"""
v15c — Full patient sweep and data quality audit.

Trains a SEPARATE model for ALL eligible patients to:
  1. Confirm the temporal signal is universal
  2. Identify noisy/corrupted patients
  3. Produce a quality JSON report for downstream filtering
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
        return self.head(h.squeeze(0)).squeeze(-1)


def classify_tier(rho):
    if rho >= 0.9:
        return "excellent"
    elif rho >= 0.7:
        return "good"
    elif rho >= 0.3:
        return "weak"
    return "poor"


def train_single_patient(data, labels, times_sec, pid, config, device):
    """Train a fresh model on one patient, return metrics dict."""
    tte_min = times_sec.float().clamp(min=0) / 60.0

    sr_mask = labels == -1
    sr_data = data[sr_mask]
    sr_tte = tte_min[sr_mask]

    if len(sr_data) < 20:
        return None

    N = len(sr_data)
    idx = torch.randperm(N)
    split = int(N * (1 - config["val_split"]))

    train_data, val_data = sr_data[idx[:split]], sr_data[idx[split:]]
    train_tte, val_tte = sr_tte[idx[:split]], sr_tte[idx[split:]]

    train_loader = DataLoader(
        TensorDataset(train_data, train_tte),
        batch_size=config["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_tte),
        batch_size=config["batch_size"], shuffle=False,
    )

    model = TTEPredictor(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01,
    )
    loss_fn = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(config["epochs"]):
        model.train()
        for rr, tte in train_loader:
            rr, tte = rr.to(device), tte.to(device)
            optimizer.zero_grad()
            pred = model(rr)
            loss = loss_fn(pred, tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        v_preds, v_actual = [], []
        with torch.no_grad():
            for rr, tte in val_loader:
                rr, tte = rr.to(device), tte.to(device)
                v_preds.append(model(rr).cpu())
                v_actual.append(tte.cpu())

        v_preds = torch.cat(v_preds).numpy()
        v_actual = torch.cat(v_actual).numpy()
        v_mae = np.mean(np.abs(v_preds - v_actual))

        if v_mae < best_val_mae:
            best_val_mae = v_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_actual = [], []
    eval_loader = DataLoader(
        TensorDataset(sr_data, sr_tte), batch_size=64, shuffle=False,
    )
    with torch.no_grad():
        for rr, tte in eval_loader:
            all_preds.append(model(rr.to(device)).cpu().numpy())
            all_actual.append(tte.numpy())

    all_preds = np.concatenate(all_preds)
    all_actual = np.concatenate(all_actual)

    mae = np.mean(np.abs(all_preds - all_actual))
    rho, _ = spearmanr(all_preds, all_actual)
    rho = float(rho) if not np.isnan(rho) else 0.0
    ss_res = np.sum((all_actual - all_preds) ** 2)
    ss_tot = np.sum((all_actual - all_actual.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    return {
        "pid": int(pid),
        "n_sr": int(len(sr_data)),
        "n_train": int(len(train_data)),
        "n_val": int(len(val_data)),
        "tte_range_min": float(sr_tte.max()),
        "mae_min": float(mae),
        "spearman": float(rho),
        "r2": float(r2),
        "tier": classify_tier(rho),
        "stopped_epoch": epoch + 1,
        "preds": all_preds,
        "actual": all_actual,
    }


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
        "epochs": 400,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "val_split": 0.2,
        "patience": 20,
        "min_sr_segments": 20,
    }

    run = wandb.init(
        entity="eml-labs", project="CPC-New-Temporal-Ranking",
        name="v15c-full-patient-audit", config=config,
    )

    torch.manual_seed(42)
    np.random.seed(42)

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
    labels_np = ds.labels.numpy()
    unique_pids, counts = np.unique(pids_np, return_counts=True)

    sr_counts = np.array([
        int((labels_np[pids_np == pid] == -1).sum()) for pid in unique_pids
    ])

    eligible = np.where(sr_counts >= config["min_sr_segments"])[0]
    n_total = len(eligible)

    logger.info(f"Full patient audit: {n_total} eligible patients "
                f"(from {len(unique_pids)} total, >= {config['min_sr_segments']} SR segments)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ══════════════════════════════════════════════════════════════════
    #  Train each patient independently
    # ══════════════════════════════════════════════════════════════════
    results = []
    pbar = tqdm(total=n_total, desc="Patient sweep")

    for rank, arr_idx in enumerate(eligible):
        pid = unique_pids[arr_idx]
        mask = pids_np == pid
        p_data = ds.data[mask]
        p_labels = ds.labels[mask]
        p_times = ds.times[mask]

        torch.manual_seed(42)
        np.random.seed(42)

        res = train_single_patient(p_data, p_labels, p_times, pid, config, device)
        pbar.update(1)

        if res is None:
            pbar.write(f"  Patient {pid}: skipped (too few SR)")
            continue

        results.append(res)

        pbar.write(
            f"  P{pid:>3} [{res['tier']:>9}] "
            f"SR={res['n_sr']:>3} | "
            f"MAE={res['mae_min']:>6.2f} | "
            f"ρ={res['spearman']:>+6.3f} | "
            f"R²={res['r2']:>+6.3f} | "
            f"@{res['stopped_epoch']}"
        )

    pbar.close()

    # ══════════════════════════════════════════════════════════════════
    #  Tier breakdown
    # ══════════════════════════════════════════════════════════════════
    tier_counts = {"excellent": 0, "good": 0, "weak": 0, "poor": 0}
    for r in results:
        tier_counts[r["tier"]] += 1

    maes = [r["mae_min"] for r in results]
    rhos = [r["spearman"] for r in results]
    r2s = [r["r2"] for r in results]

    logger.info(f"\n{'='*70}")
    logger.info("FULL PATIENT AUDIT — Per-Patient TTE Prediction (separate models)")
    logger.info(f"{'='*70}")
    logger.info(f"  Total patients tested: {len(results)}")
    logger.info(f"  Excellent (ρ≥0.9): {tier_counts['excellent']}")
    logger.info(f"  Good  (0.7≤ρ<0.9): {tier_counts['good']}")
    logger.info(f"  Weak  (0.3≤ρ<0.7): {tier_counts['weak']}")
    logger.info(f"  Poor     (ρ<0.3):   {tier_counts['poor']}")
    logger.info(f"{'─'*70}")
    logger.info(f"  Mean ρ:   {np.mean(rhos):+.3f} ± {np.std(rhos):.3f}")
    logger.info(f"  Mean MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f} min")
    logger.info(f"  Mean R²:  {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    logger.info(f"{'─'*70}")

    logger.info(f"\n{'PID':>5} {'Tier':>10} {'SR':>5} {'Range':>8} {'MAE':>8} {'ρ':>8} {'R²':>8} {'Ep':>4}")
    logger.info(f"{'─'*62}")
    for r in sorted(results, key=lambda x: x["spearman"], reverse=True):
        logger.info(
            f"{r['pid']:>5} {r['tier']:>10} {r['n_sr']:>5} "
            f"{r['tte_range_min']:>7.1f}m "
            f"{r['mae_min']:>7.2f}m "
            f"{r['spearman']:>+7.3f} "
            f"{r['r2']:>7.3f} "
            f"{r['stopped_epoch']:>4}"
        )
    logger.info(f"{'='*70}")

    # ══════════════════════════════════════════════════════════════════
    #  Save JSON quality report (without numpy arrays)
    # ══════════════════════════════════════════════════════════════════
    json_results = []
    for r in sorted(results, key=lambda x: x["spearman"], reverse=True):
        json_results.append({
            "pid": r["pid"],
            "n_sr": r["n_sr"],
            "tte_range_min": round(r["tte_range_min"], 2),
            "mae_min": round(r["mae_min"], 2),
            "spearman": round(r["spearman"], 4),
            "r2": round(r["r2"], 4),
            "tier": r["tier"],
        })

    with open("v15c_patient_quality.json", "w") as f:
        json.dump({
            "summary": {
                "n_patients": len(results),
                "tiers": tier_counts,
                "mean_spearman": round(float(np.mean(rhos)), 4),
                "std_spearman": round(float(np.std(rhos)), 4),
                "mean_mae": round(float(np.mean(maes)), 2),
                "mean_r2": round(float(np.mean(r2s)), 4),
            },
            "patients": json_results,
        }, f, indent=2)
    logger.info("Quality report saved to v15c_patient_quality.json")

    run.log({
        "summary/n_patients": len(results),
        "summary/n_excellent": tier_counts["excellent"],
        "summary/n_good": tier_counts["good"],
        "summary/n_weak": tier_counts["weak"],
        "summary/n_poor": tier_counts["poor"],
        "summary/mean_mae": np.mean(maes),
        "summary/mean_spearman": np.mean(rhos),
        "summary/mean_r2": np.mean(r2s),
        "summary/std_spearman": np.std(rhos),
    })

    # ══════════════════════════════════════════════════════════════════
    #  Plots — Row 1: distributions, Row 2: best 3 + worst 3 scatters
    # ══════════════════════════════════════════════════════════════════
    sorted_by_rho = sorted(results, key=lambda x: x["spearman"], reverse=True)
    best3 = sorted_by_rho[:3]
    worst3 = sorted_by_rho[-3:]

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.4)

    # Row 1, Col 1-2: ρ histogram
    ax = fig.add_subplot(gs[0, 0:2])
    ax.hist(rhos, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(rhos), color="k", ls="--", lw=1.5, label=f"mean={np.mean(rhos):.3f}")
    ax.axvline(0.9, color="green", ls=":", lw=1, label="excellent (0.9)")
    ax.axvline(0.3, color="red", ls=":", lw=1, label="poor (0.3)")
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Count")
    ax.set_title(f"ρ Distribution ({len(results)} patients)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1, Col 3-4: R² histogram
    ax = fig.add_subplot(gs[0, 2:4])
    ax.hist(r2s, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(r2s), color="k", ls="--", lw=1.5, label=f"mean={np.mean(r2s):.3f}")
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("R² Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1, Col 5-6: ρ vs n_sr scatter
    ax = fig.add_subplot(gs[0, 4:6])
    n_srs = [r["n_sr"] for r in results]
    tier_color = {"excellent": "steelblue", "good": "green", "weak": "orange", "poor": "red"}
    for r in results:
        ax.scatter(r["n_sr"], r["spearman"], c=tier_color[r["tier"]], s=30, alpha=0.7)
    for tier, color in tier_color.items():
        ax.scatter([], [], c=color, label=tier, s=30)
    ax.set_xlabel("Number of SR segments")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("ρ vs Data Size")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # Row 2: best 3 + worst 3 actual vs predicted
    showcase = best3 + worst3
    labels_row2 = ["BEST"] * 3 + ["WORST"] * 3
    for i, (r, tag) in enumerate(zip(showcase, labels_row2)):
        ax = fig.add_subplot(gs[1, i])
        ax.scatter(r["actual"], r["preds"], alpha=0.5, s=12,
                   c="steelblue" if tag == "BEST" else "coral")
        lim = max(r["actual"].max(), r["preds"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(min(0, r["preds"].min()) - 1, lim)
        ax.set_aspect("equal")
        ax.set_title(
            f"{tag} P{r['pid']} [{r['tier']}]\n"
            f"ρ={r['spearman']:.3f}  R²={r['r2']:.3f}  SR={r['n_sr']}",
            fontsize=9,
        )
        ax.set_xlabel("Actual (min)", fontsize=7)
        ax.set_ylabel("Predicted (min)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"v15c Full Patient Audit  |  {len(results)} patients  |  "
        f"Excellent={tier_counts['excellent']}  Good={tier_counts['good']}  "
        f"Weak={tier_counts['weak']}  Poor={tier_counts['poor']}  |  "
        f"mean ρ={np.mean(rhos):.3f}",
        fontsize=14,
    )

    plt.savefig("v15c_patient_audit.png", dpi=150, bbox_inches="tight")
    run.log({"final_plot": wandb.Image(fig)})
    plt.close(fig)

    logger.info("Plot saved to v15c_patient_audit.png")

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
finally:
    if "run" in locals():
        run.finish()
