"""
v15b — Per-patient TTE prediction across 10 patients.

Trains a SEPARATE model for each of the top-10 patients (by SR segment count)
to verify whether the temporal signal exists universally or only in some patients.
Same architecture and training as v15, just looped.
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
        return self.head(h.squeeze(0)).squeeze(-1)


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
    rho = rho if not np.isnan(rho) else 0.0
    ss_res = np.sum((all_actual - all_preds) ** 2)
    ss_tot = np.sum((all_actual - all_actual.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    return {
        "pid": int(pid),
        "n_sr": len(sr_data),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "tte_range_min": float(sr_tte.max()),
        "mae_min": mae,
        "spearman": rho,
        "r2": r2,
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
        "epochs": 200,
        "batch_size": 32,
        "lr": 1e-3,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "val_split": 0.2,
        "patience": 25,
        "n_patients": 10,
    }

    run = wandb.init(
        entity="eml-labs", project="CPC-New-Temporal-Ranking",
        name="v15b-per-patient-sweep", config=config,
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

    # Filter to patients with enough SR segments, then sample randomly
    eligible = np.where(sr_counts >= 20)[0]
    np.random.shuffle(eligible)
    top_indices = eligible[:config["n_patients"]]

    logger.info(f"Training separate models for {len(top_indices)} randomly selected patients "
                f"(from {len(eligible)} eligible with >=20 SR segments):")
    for i in top_indices:
        pid = unique_pids[i]
        logger.info(f"  Patient {pid}: {counts[i]} total, {sr_counts[i]} SR")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ══════════════════════════════════════════════════════════════════
    #  Train each patient independently
    # ══════════════════════════════════════════════════════════════════
    results = []

    for rank, arr_idx in enumerate(top_indices):
        pid = unique_pids[arr_idx]
        mask = pids_np == pid
        p_data = ds.data[mask]
        p_labels = ds.labels[mask]
        p_times = ds.times[mask]

        logger.info(f"\n{'─'*50}")
        logger.info(f"[{rank+1}/{config['n_patients']}] Patient {pid}")

        torch.manual_seed(42)
        np.random.seed(42)

        res = train_single_patient(p_data, p_labels, p_times, pid, config, device)
        if res is None:
            logger.info(f"  Skipped (too few SR segments)")
            continue

        results.append(res)

        logger.info(
            f"  SR={res['n_sr']} | "
            f"MAE={res['mae_min']:.2f} min | "
            f"ρ={res['spearman']:.3f} | "
            f"R²={res['r2']:.3f} | "
            f"range=0–{res['tte_range_min']:.0f} min | "
            f"stopped@{res['stopped_epoch']}"
        )

        run.log({
            f"patient_{pid}/mae": res["mae_min"],
            f"patient_{pid}/spearman": res["spearman"],
            f"patient_{pid}/r2": res["r2"],
            f"patient_{pid}/n_sr": res["n_sr"],
        })

    # ══════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*65}")
    logger.info("SUMMARY — Per-Patient TTE Prediction (separate models)")
    logger.info(f"{'='*65}")
    logger.info(f"{'PID':>5} {'SR':>5} {'Range':>8} {'MAE':>8} {'ρ':>8} {'R²':>8} {'Epoch':>6}")
    logger.info(f"{'─'*55}")

    maes, rhos, r2s = [], [], []
    for r in sorted(results, key=lambda x: x["spearman"], reverse=True):
        logger.info(
            f"{r['pid']:>5} {r['n_sr']:>5} "
            f"{r['tte_range_min']:>7.1f}m "
            f"{r['mae_min']:>7.2f}m "
            f"{r['spearman']:>+7.3f} "
            f"{r['r2']:>7.3f} "
            f"{r['stopped_epoch']:>5}"
        )
        maes.append(r["mae_min"])
        rhos.append(r["spearman"])
        r2s.append(r["r2"])

    logger.info(f"{'─'*55}")
    logger.info(
        f"{'MEAN':>5} {'':>5} {'':>8} "
        f"{np.mean(maes):>7.2f}m "
        f"{np.mean(rhos):>+7.3f} "
        f"{np.mean(r2s):>7.3f}"
    )
    logger.info(
        f"{'STD':>5} {'':>5} {'':>8} "
        f"{np.std(maes):>7.2f}m "
        f"{np.std(rhos):>7.3f} "
        f"{np.std(r2s):>7.3f}"
    )
    logger.info(f"{'='*65}")

    run.log({
        "summary/mean_mae": np.mean(maes),
        "summary/mean_spearman": np.mean(rhos),
        "summary/mean_r2": np.mean(r2s),
        "summary/std_spearman": np.std(rhos),
        "summary/n_patients": len(results),
    })

    # ══════════════════════════════════════════════════════════════════
    #  Plots — 2 rows: top = summary bars, bottom = per-patient scatters
    # ══════════════════════════════════════════════════════════════════
    n = len(results)
    fig = plt.figure(figsize=(max(24, n * 2.5), 12))

    gs = fig.add_gridspec(2, max(n, 3), hspace=0.35, wspace=0.35)

    # Row 1, Col 1: Spearman ρ bar chart
    ax = fig.add_subplot(gs[0, 0])
    pids_sorted = [r["pid"] for r in sorted(results, key=lambda x: x["spearman"])]
    rhos_sorted = [r["spearman"] for r in sorted(results, key=lambda x: x["spearman"])]
    colors = ["steelblue" if r >= 0.5 else "orange" if r >= 0 else "coral" for r in rhos_sorted]
    ax.barh(range(n), rhos_sorted, color=colors, height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"P{p}" for p in pids_sorted], fontsize=8)
    ax.axvline(np.mean(rhos), color="k", ls="--", lw=1, label=f"mean={np.mean(rhos):.3f}")
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Per-Patient ρ")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1, Col 2: MAE bar chart
    ax = fig.add_subplot(gs[0, 1])
    pids_mae = [r["pid"] for r in sorted(results, key=lambda x: x["mae_min"])]
    maes_sorted = [r["mae_min"] for r in sorted(results, key=lambda x: x["mae_min"])]
    ax.barh(range(n), maes_sorted, color="steelblue", height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"P{p}" for p in pids_mae], fontsize=8)
    ax.axvline(np.mean(maes), color="k", ls="--", lw=1, label=f"mean={np.mean(maes):.2f}")
    ax.set_xlabel("MAE (min)")
    ax.set_title("Per-Patient MAE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1, Col 3: R² bar chart
    ax = fig.add_subplot(gs[0, 2])
    pids_r2 = [r["pid"] for r in sorted(results, key=lambda x: x["r2"])]
    r2s_sorted = [r["r2"] for r in sorted(results, key=lambda x: x["r2"])]
    r2_colors = ["steelblue" if r >= 0.5 else "orange" if r >= 0 else "coral" for r in r2s_sorted]
    ax.barh(range(n), r2s_sorted, color=r2_colors, height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"P{p}" for p in pids_r2], fontsize=8)
    ax.axvline(np.mean(r2s), color="k", ls="--", lw=1, label=f"mean={np.mean(r2s):.3f}")
    ax.set_xlabel("R²")
    ax.set_title("Per-Patient R²")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 2: per-patient actual vs predicted scatter
    for i, r in enumerate(sorted(results, key=lambda x: x["spearman"], reverse=True)):
        ax = fig.add_subplot(gs[1, i])
        ax.scatter(r["actual"], r["preds"], alpha=0.5, s=10, c="steelblue")
        lim = max(r["actual"].max(), r["preds"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(f"P{r['pid']}  ρ={r['spearman']:.2f}  R²={r['r2']:.2f}", fontsize=9)
        ax.set_xlabel("Actual (min)", fontsize=7)
        ax.set_ylabel("Predicted (min)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"v15b Per-Patient TTE  |  {n} patients  |  "
        f"mean ρ={np.mean(rhos):.3f}  |  mean MAE={np.mean(maes):.2f} min  |  "
        f"mean R²={np.mean(r2s):.3f}",
        fontsize=14,
    )

    plt.savefig("v15b_per_patient_results.png", dpi=150, bbox_inches="tight")
    run.log({"final_plot": wandb.Image(fig)})
    plt.close(fig)

    logger.info("Plot saved to v15b_per_patient_results.png")

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
finally:
    if "run" in locals():
        run.finish()
