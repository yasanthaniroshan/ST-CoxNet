"""
v17 patient identity probe

Goal: check whether the trained TTE model embeddings encode patient identity.

Protocol:
1) Train v17-style TTE regressor on SR segments filtered to v15c tier=="excellent"
2) Use a segment-level random split (intentional leakage allowed; probe focuses on
   representation content, not strict generalization)
3) Freeze the regressor and extract frozen GRU embeddings (h) for each segment
4) Train a simple linear classifier to predict patient_id from frozen embeddings
5) Evaluate classifier accuracy on the held-out segment split

If probe accuracy is far above random baseline (1/C), the representation strongly
encodes patient identity.
"""

import json
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset
from scipy.stats import spearmanr

import wandb
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TTEPredictor(nn.Module):
    """Encoder → GRU → regression head (TTE minutes)."""

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
        """Return TTE prediction for a batch of segments."""
        B, T, W = rr_windows.shape
        z = self.encoder(rr_windows.view(B * T, W)).view(B, T, -1)
        _, h = self.gru(z)
        return self.head(h.squeeze(0)).squeeze(-1)  # [B]

    @torch.no_grad()
    def extract_gru_embedding(self, rr_windows):
        """Return GRU hidden state embedding h for each segment: [B, hidden_dim]."""
        B, T, W = rr_windows.shape
        z = self.encoder(rr_windows.view(B * T, W)).view(B, T, -1)
        _, h = self.gru(z)
        return h.squeeze(0)  # [B, hidden_dim]


def build_excellent_patient_set(v15c_quality_path):
    with open(v15c_quality_path, "r") as f:
        quality = json.load(f)

    excellent_pids = set()
    for p in quality["patients"]:
        if p.get("tier") == "excellent":
            excellent_pids.add(int(p["pid"]))
    return excellent_pids, quality


def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()


try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    config = {
        # Dataset
        "afib_length": 60 * 60,
        "sr_length": int(1.5 * 60 * 60),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,  # segment-level split for both regressor + probe
        # TTE regressor training
        "epochs": 200,
        "batch_size": 64,
        "lr": 1e-3,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "patience": 20,
        # Probe training
        "probe_epochs": 50,
        "probe_lr": 1e-3,
        "min_classes": 2,
        "seed": 42,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    project_dir = os.path.dirname(__file__)
    v15c_quality_path = os.path.join(project_dir, "v15c_patient_quality.json")
    if not os.path.exists(v15c_quality_path):
        raise FileNotFoundError(f"Missing {v15c_quality_path}. Run v15c first.")

    excellent_pids, quality = build_excellent_patient_set(v15c_quality_path)

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name="v17-patient-identity-probe",
        config=config,
    )

    logger.info(
        f"Excellent-only patient set: {len(excellent_pids)} / {quality['summary']['n_patients']} total"
    )

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

    # Pool train+val segments (so the regressor sees excellent patients with leakage allowed)
    all_data = torch.cat([train_ds_full.data, val_ds_full.data], dim=0)
    all_times_sec = torch.cat([train_ds_full.times, val_ds_full.times], dim=0)
    all_pids = torch.cat([train_ds_full.patient_ids, val_ds_full.patient_ids], dim=0)

    all_pids_np = all_pids.numpy()
    mask_excellent = np.isin(all_pids_np, list(excellent_pids))

    all_data = all_data[mask_excellent]
    all_times_sec = all_times_sec[mask_excellent]
    all_pids = all_pids[mask_excellent]

    # Segment-level random split indices (intentional leakage allowed)
    N = len(all_data)
    idx = torch.randperm(N)
    split = int(N * (1 - config["validation_split"]))
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_data = all_data[train_idx]
    val_data = all_data[val_idx]
    train_tte_min = all_times_sec[train_idx].float().clamp(min=0) / 60.0
    val_tte_min = all_times_sec[val_idx].float().clamp(min=0) / 60.0
    train_pids = all_pids[train_idx]
    val_pids = all_pids[val_idx]

    logger.info(
        f"TTE regressor train/val segments (excellent only): "
        f"{len(train_data)}/{len(val_data)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictor(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01
    )
    loss_fn = nn.SmoothL1Loss()

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

    best_val_mae = float("inf")
    patience_ctr = 0
    best_state = None

    reg_pbar = tqdm(total=config["epochs"], desc="Regressor")
    for epoch in range(config["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for rr, tte, _ in train_loader:
            rr, tte = rr.to(device), tte.to(device)
            optimizer.zero_grad()
            pred = model(rr)
            loss = loss_fn(pred, tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * rr.size(0)
            train_n += rr.size(0)
        scheduler.step()

        model.eval()
        v_preds, v_actual, v_pids = [], [], []
        with torch.no_grad():
            for rr, tte, pid in val_loader:
                rr = rr.to(device)
                pred = model(rr)
                v_preds.append(pred.cpu().numpy())
                v_actual.append(tte.numpy())
                v_pids.append(pid.numpy())

        v_preds = np.concatenate(v_preds)
        v_actual = np.concatenate(v_actual)
        v_pids = np.concatenate(v_pids)

        v_mae = float(np.mean(np.abs(v_preds - v_actual)))
        v_rho, _ = spearmanr(v_preds, v_actual)
        v_rho = 0.0 if np.isnan(v_rho) else float(v_rho)

        # Patient mean rho
        per_patient = {}
        for pid in np.unique(v_pids):
            m = v_pids == pid
            if m.sum() < 10:
                continue
            pp_rho, _ = spearmanr(v_preds[m], v_actual[m])
            if np.isnan(pp_rho):
                continue
            per_patient[int(pid)] = float(pp_rho)
        mean_pp_rho = float(np.mean(list(per_patient.values()))) if per_patient else 0.0

        run.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_sum / max(train_n, 1),
            "val/mae_min": v_mae,
            "val/spearman_global": v_rho,
            "val/spearman_per_patient": mean_pp_rho,
        })

        reg_pbar.update(1)
        reg_pbar.set_postfix({
            "vMAE": f"{v_mae:.2f}",
            "vRho": f"{v_rho:.3f}",
            "vP-Rho": f"{mean_pp_rho:.3f}",
        })
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Regressor epoch {epoch+1:>3}/{config['epochs']} | "
                f"train_loss={train_loss_sum / max(train_n,1):.4f} | "
                f"val_MAE={v_mae:.2f} | val_rho={v_rho:.3f} | "
                f"val_patient_rho={mean_pp_rho:.3f}"
            )

        if v_mae < best_val_mae:
            best_val_mae = v_mae
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                logger.info(f"Early stop TTE regressor at epoch {epoch+1}")
                break
    reg_pbar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ────────────────────────────────────────────────────────────────
    # Extract frozen GRU embeddings for probe
    # ────────────────────────────────────────────────────────────────
    def embed_all(loader):
        embs, pids_all, tte_all = [], [], []
        with torch.no_grad():
            for rr, tte, pid in loader:
                rr = rr.to(device)
                emb = model.extract_gru_embedding(rr)  # [B, hidden_dim]
                embs.append(emb.cpu().numpy())
                pids_all.append(pid.numpy())
                tte_all.append(tte.numpy())
        return (
            np.concatenate(embs),
            np.concatenate(pids_all),
            np.concatenate(tte_all),
        )

    train_emb, train_pid_probe, _ = embed_all(train_loader)
    val_emb, val_pid_probe, _ = embed_all(val_loader)

    all_probe_pids = np.unique(np.concatenate([train_pid_probe, val_pid_probe]))
    if len(all_probe_pids) < config["min_classes"]:
        raise RuntimeError("Not enough patient classes for probe.")

    pid_to_class = {pid: i for i, pid in enumerate(sorted(all_probe_pids))}
    train_y = np.array([pid_to_class[int(pid)] for pid in train_pid_probe], dtype=np.int64)
    val_y = np.array([pid_to_class[int(pid)] for pid in val_pid_probe], dtype=np.int64)

    n_classes = len(pid_to_class)
    random_baseline = 1.0 / n_classes
    logger.info(f"Probe classes: {n_classes} | random baseline: {random_baseline:.4f}")

    # Normalize embeddings (train stats only)
    tr_mean = train_emb.mean(axis=0, keepdims=True)
    tr_std = train_emb.std(axis=0, keepdims=True) + 1e-8
    train_emb_n = (train_emb - tr_mean) / tr_std
    val_emb_n = (val_emb - tr_mean) / tr_std

    probe_train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_emb_n, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.long),
        ),
        batch_size=256, shuffle=True,
    )
    probe_val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_emb_n, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.long),
        ),
        batch_size=256, shuffle=False,
    )

    probe = nn.Linear(config["hidden_dim"], n_classes).to(device)
    probe_opt = optim.AdamW(probe.parameters(), lr=config["probe_lr"], weight_decay=1e-4)
    probe_loss_fn = nn.CrossEntropyLoss()

    best_probe_acc = 0.0
    probe_pbar = tqdm(total=config["probe_epochs"], desc="Probe")
    for ep in range(config["probe_epochs"]):
        probe.train()
        for x, y in probe_train_loader:
            x, y = x.to(device), y.to(device)
            probe_opt.zero_grad()
            logits = probe(x)
            loss = probe_loss_fn(logits, y)
            loss.backward()
            probe_opt.step()

        probe.eval()
        with torch.no_grad():
            all_logits, all_targets = [], []
            for x, y in probe_val_loader:
                x = x.to(device)
                logits = probe(x)
                all_logits.append(logits.cpu())
                all_targets.append(y)
            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            val_acc = compute_accuracy(all_logits, all_targets)

        best_probe_acc = max(best_probe_acc, val_acc)
        run.log({
            "probe/epoch": ep + 1,
            "probe/val_accuracy": val_acc,
            "probe/random_baseline": random_baseline,
        })
        probe_pbar.update(1)
        probe_pbar.set_postfix({
            "val_acc": f"{val_acc:.3f}",
            "best": f"{best_probe_acc:.3f}",
        })
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(
                f"Probe epoch {ep+1:>3}/{config['probe_epochs']} | "
                f"val_acc={val_acc:.3f} | best={best_probe_acc:.3f} | "
                f"random={random_baseline:.3f}"
            )
    probe_pbar.close()

    # ────────────────────────────────────────────────────────────────
    # Save probe results
    # ────────────────────────────────────────────────────────────────
    probe_results = {
        "n_classes": int(n_classes),
        "random_baseline": float(random_baseline),
        "best_probe_val_accuracy": float(best_probe_acc),
        "val_emb_shape": list(val_emb_n.shape),
        "seed": int(config["seed"]),
        "excellent_patients_used": len(excellent_pids),
        "note": "Segment-level split; probe accuracy reflects patient identity encoding, not strict generalization.",
    }

    out_path = os.path.join(project_dir, "v17_patient_identity_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(probe_results, f, indent=2)

    logger.info("Probe results saved to v17_patient_identity_probe_results.json")
    logger.info(
        f"Best probe val accuracy: {best_probe_acc:.4f} vs random baseline {random_baseline:.4f}"
    )

    run.log({"probe/best_val_accuracy": best_probe_acc})

    # Optional quick visualization: 2D PCA of embeddings colored by class index
    # (Keep it lightweight: if too many classes, plot only a subset.)
    try:
        from sklearn.decomposition import PCA

        max_classes_to_plot = 12
        class_ids = sorted(pid_to_class.values())
        if n_classes > max_classes_to_plot:
            chosen = set(np.random.choice(class_ids, size=max_classes_to_plot, replace=False))
            plot_mask = np.isin(val_y, list(chosen))
        else:
            plot_mask = np.ones_like(val_y, dtype=bool)

        x2 = PCA(n_components=2, random_state=config["seed"]).fit_transform(val_emb_n[plot_mask])
        y2 = val_y[plot_mask]

        plt.figure(figsize=(8, 6))
        plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=8, alpha=0.7, cmap="tab10")
        plt.title("Probe embeddings PCA (subset of classes)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        pth = os.path.join(project_dir, "v17_probe_embedding_pca.png")
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        run.log({"probe/pca_plot": wandb.Image(plt)})
        plt.close()
    except Exception as _:
        pass

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
finally:
    if "run" in locals():
        run.finish()

