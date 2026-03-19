"""
CPC v10: Multi-modal time-bin classification
=============================================
Key insight from debug_time_signal.py:
  - RF on HRV features gets 82.2% (vs NN 38.4%, random 33.3%)
  - The temporal signal EXISTS; the NN fails to extract it from raw RR
  - HRV features are far more discriminative than raw flattened RR

This version:
  1. Pre-computes HRV features for all segments (cached to disk)
  2. Loads CPC checkpoint from v8 (BaseEncoder)
  3. Multi-modal classification: CPC latent/context + HRV features -> TimeBinHead
  4. Optional: ResNetEncoder swap (requires CPC retraining from scratch)
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import wandb

import json
import hashlib
import h5py

from Model.CPC import CPC
from Utils.Dataset.CPCDataset import CPCDataset
from Utils.Dataset.CPCTimeBinDataset import CPCTimeBinDataset

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

BIN_NAMES = ["Near (0-33%)", "Mid (33-67%)", "Far (67-100%)"]


# ======================================================================
# HRV Feature Computation (same as debug_time_signal.py)
# ======================================================================


def filter_segment(seg_flat, iqr_factor=5.0):
    q1, q3 = np.percentile(seg_flat, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-8:
        return seg_flat
    lo = q1 - iqr_factor * iqr
    hi = q3 + iqr_factor * iqr
    mask = (seg_flat >= lo) & (seg_flat <= hi)
    return seg_flat[mask]


def _dfa_alpha1(rr, scales=None):
    N = len(rr)
    if N < 16:
        return np.nan
    y = np.cumsum(rr - np.mean(rr))
    if scales is None:
        scales = np.arange(4, min(12, N // 4 + 1))
    if len(scales) < 2:
        return np.nan
    flucts = []
    for n in scales:
        segs = N // n
        if segs == 0:
            flucts.append(np.nan)
            continue
        rms_vals = []
        for i in range(segs):
            seg = y[i * n : (i + 1) * n]
            x = np.arange(n)
            poly = np.polyfit(x, seg, 1)
            trend = np.polyval(poly, x)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        flucts.append(np.mean(rms_vals))
    flucts = np.array(flucts)
    valid = ~np.isnan(flucts) & (flucts > 0)
    if valid.sum() < 2:
        return np.nan
    return np.polyfit(np.log(scales[valid]), np.log(flucts[valid]), 1)[0]


def _sample_entropy(rr, m=2, r_frac=0.2, max_len=100):
    if len(rr) > max_len:
        rr = rr[:max_len]
    N = len(rr)
    if N <= m + 1:
        return np.nan
    r = r_frac * np.std(rr)
    if r < 1e-10:
        return np.nan

    def _count_full(data, m_val, r_val):
        n = len(data) - m_val
        if n < 2:
            return 0
        idx = np.arange(m_val)[None, :] + np.arange(n)[:, None]
        templates = data[idx]
        dist = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        tri = np.triu_indices(n, k=1)
        return int(np.sum(dist[tri] <= r_val))

    A = _count_full(rr, m + 1, r)
    B = _count_full(rr, m, r)
    if B == 0:
        return np.nan
    if A == 0:
        return -np.log(1.0 / (B + 1e-6))
    return float(-np.log(A / B))


def compute_hrv_features(seg_flat_raw):
    """15 HRV features from a single flattened segment (RobustScaler-transformed)."""
    filtered = filter_segment(seg_flat_raw)
    artifact_rate = 1.0 - len(filtered) / max(len(seg_flat_raw), 1)
    rr = filtered if len(filtered) > 10 else seg_flat_raw

    diffs = np.diff(rr)
    feats = np.array([
        artifact_rate,
        np.mean(rr),
        np.std(rr),
        np.median(rr),
        np.percentile(rr, 75) - np.percentile(rr, 25),
        np.percentile(rr, 5),
        np.percentile(rr, 95),
        float(np.mean(((rr - np.mean(rr)) / (np.std(rr) + 1e-8)) ** 3)),
        float(np.mean(((rr - np.mean(rr)) / (np.std(rr) + 1e-8)) ** 4)),
        np.std(rr) / (np.abs(np.mean(rr)) + 1e-8),
        np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else 0.0,
        np.std(diffs) if len(diffs) > 0 else 0.0,
        np.mean(np.abs(diffs) > 0.5) if len(diffs) > 0 else 0.0,
        _dfa_alpha1(rr),
        _sample_entropy(rr, m=2, r_frac=0.2),
    ], dtype=np.float64)
    return feats


HRV_FEATURE_DIM = 15
HRV_FEATURE_NAMES = [
    "artifact_rate", "mean", "std", "median", "iqr", "p5", "p95",
    "skewness", "kurtosis", "cv", "rmssd", "sdsd", "pnn50_scaled",
    "alpha1", "sample_entropy",
]


def precompute_hrv(data_np, cache_path):
    """Compute HRV features for all segments, with caching."""
    if os.path.exists(cache_path):
        logger.info(f"Loading cached HRV features from {cache_path}")
        return np.load(cache_path)["X"]

    logger.info(f"Computing HRV features for {len(data_np)} segments...")
    n = len(data_np)
    X = np.empty((n, HRV_FEATURE_DIM), dtype=np.float64)
    for i in range(n):
        X[i] = compute_hrv_features(data_np[i].flatten())
        if (i + 1) % 1000 == 0:
            logger.info(f"  {i+1}/{n}")

    nan_mask = np.isnan(X)
    for col in range(X.shape[1]):
        col_nan = nan_mask[:, col]
        if col_nan.any():
            X[col_nan, col] = np.nanmedian(X[:, col])

    np.savez_compressed(cache_path, X=X)
    logger.info(f"  Saved {cache_path}, shape={X.shape}")
    return X


# ======================================================================
# Patient ID Recovery & Per-Patient Normalization
# ======================================================================


def _get_dataset_hash(config):
    dataset_prop = {
        "dataset_name": "IRIDIA AFIB Dataset",
        "AFIB_length_seconds": config["afib_length"],
        "SR_length_seconds": config["sr_length"],
        "window_size": config["window_size"],
        "segment_size": config["number_of_windows_in_segment"] * config["window_size"],
        "stride": config["stride"],
        "validation_split": config["validation_split"],
        "scaler": "RobustScaler",
    }
    return hashlib.sha256(json.dumps(dataset_prop, sort_keys=True).encode()).hexdigest()[:32]


def recover_patient_ids(config, processed_path, split="train"):
    """Detect patient boundaries from the monotonically-decreasing SR times."""
    h = _get_dataset_hash(config)
    h5_path = os.path.join(processed_path, f"{h}_{split}.h5")
    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:]
        times = f["times"][:]
    sr_mask = labels == -1
    sr_times = times[sr_mask]
    n = len(sr_times)
    patient_ids = np.zeros(n, dtype=np.int32)
    pid = 0
    for i in range(1, n):
        if sr_times[i] > sr_times[i - 1] + 1e-6:
            pid += 1
        patient_ids[i] = pid
    return patient_ids, pid + 1


def normalize_per_patient(X, patient_ids, n_patients):
    """Z-score normalize features within each patient."""
    X_norm = np.copy(X)
    for pid in range(n_patients):
        mask = patient_ids == pid
        if mask.sum() < 2:
            continue
        mu = X[mask].mean(axis=0)
        sigma = X[mask].std(axis=0)
        sigma[sigma < 1e-10] = 1.0
        X_norm[mask] = (X[mask] - mu) / sigma
    nan_mask = np.isnan(X_norm)
    X_norm[nan_mask] = 0.0
    return X_norm


# ======================================================================
# Multi-Modal Classification Head
# ======================================================================


class MultiModalTimeBinHead(nn.Module):
    """Concatenates CPC context + latent + HRV features for classification."""

    def __init__(self, context_dim, latent_dim, hrv_dim, num_classes, dropout=0.2):
        super().__init__()
        self.hrv_proj = nn.Sequential(
            nn.Linear(hrv_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_dim = context_dim + latent_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, context, embedding, hrv_features):
        h = self.hrv_proj(hrv_features)
        x = torch.cat([context, embedding, h], dim=1)
        return self.classifier(x)


# ======================================================================
# Utilities
# ======================================================================


def extract_features(cpc, rr):
    b, t, w = rr.shape
    z_seq = cpc.encoder(rr.view(-1, w)).view(b, t, -1)
    z_seq = F.normalize(z_seq, dim=-1)
    c_seq = cpc.ar_block(z_seq)
    return z_seq, c_seq


def compute_multiclass_metrics(y_true, y_pred, num_classes):
    acc = float((y_pred == y_true).mean()) if len(y_true) > 0 else 0.0
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return acc, f1m, f1w, prec, rec, cm


def plot_confusion_matrix(cm, class_names, epoch):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (epoch {epoch})")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_per_class_acc(cm, class_names, epoch):
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(class_names)), per_class, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Class Accuracy (epoch {epoch})")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    return fig


def plot_curves(epochs, train_vals, val_vals, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_vals, label="Train")
    ax.plot(epochs, val_vals, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    return fig


# ======================================================================
# Main
# ======================================================================

try:
    config = {
        "afib_length": 60 * 60,
        "sr_length": int(1.5 * 60 * 60),
        "number_of_windows_in_segment": 10,
        "stride": 20,
        "window_size": 100,
        "validation_split": 0.15,
        "dropout": 0.1,
        "temperature": 0.07,
        "latent_dim": 64,
        "context_dim": 128,
        "number_of_prediction_steps": 3,
        "bin_edges": [0.0, 1 / 3, 2 / 3, 1.01],
        # CPC (reuse v8 checkpoint by default)
        "cpc_epochs": 50,
        "cpc_batch_size": 512,
        "cpc_lr": 1e-3,
        "cpc_patience": 8,
        # Multi-modal classification
        "cls_epochs": 60,
        "cls_batch_size": 256,
        "cls_lr": 3e-4,
        "cls_backbone_lr_scale": 0.01,
        "cls_head_warmup_epochs": 12,
        "cls_patience": 12,
        "hrv_dim": HRV_FEATURE_DIM,
    }

    run = wandb.init(entity="eml-labs", project="CPC-MultiModal-PatientNorm-v10", config=config)

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/debug_signal", exist_ok=True)

    processed_dataset_path = "/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =================================================================
    # STAGE 1: CPC — load v8 checkpoint or train
    # =================================================================
    cpc = CPC(
        latent_dim=config["latent_dim"],
        context_dim=config["context_dim"],
        number_of_prediction_steps=config["number_of_prediction_steps"],
        dropout=config["dropout"],
        temperature=config["temperature"],
        window_size=config["window_size"],
    ).to(device)

    cpc_ckpt = "cpc_model_v8.pth"
    if os.path.exists(cpc_ckpt):
        logger.info(f"Loading CPC checkpoint from {cpc_ckpt} (skipping pretraining)")
        cpc.load_state_dict(torch.load(cpc_ckpt, weights_only=True))
    else:
        logger.info("=== STAGE 1: CPC Pretraining ===")
        cpc_train_ds = CPCDataset(
            processed_dataset_path=processed_dataset_path,
            afib_length=config["afib_length"], sr_length=config["sr_length"],
            number_of_windows_in_segment=config["number_of_windows_in_segment"],
            stride=config["stride"], window_size=config["window_size"],
            validation_split=config["validation_split"], train=True,
        )
        cpc_val_ds = CPCDataset(
            processed_dataset_path=processed_dataset_path,
            afib_length=config["afib_length"], sr_length=config["sr_length"],
            number_of_windows_in_segment=config["number_of_windows_in_segment"],
            stride=config["stride"], window_size=config["window_size"],
            validation_split=config["validation_split"], train=False,
        )
        logger.info(f"CPC train: {len(cpc_train_ds)}, val: {len(cpc_val_ds)}")

        cpc_train_loader = DataLoader(
            cpc_train_ds, batch_size=config["cpc_batch_size"], shuffle=True,
            drop_last=True, num_workers=8, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )
        cpc_val_loader = DataLoader(
            cpc_val_ds, batch_size=config["cpc_batch_size"], shuffle=False,
            drop_last=False, num_workers=8, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )
        cpc_opt = optim.AdamW(cpc.parameters(), lr=config["cpc_lr"], weight_decay=1e-2)
        cpc_sched = optim.lr_scheduler.CosineAnnealingLR(
            cpc_opt, T_max=config["cpc_epochs"], eta_min=config["cpc_lr"] * 0.1,
        )
        best_cpc_val_acc = 0.0
        cpc_patience_ctr = 0

        pbar = tqdm(total=config["cpc_epochs"], desc="Stage 1: CPC")
        for ep in range(config["cpc_epochs"]):
            cpc.train()
            t_loss, t_acc, t_n = 0.0, 0.0, 0
            for rr, _, _ in cpc_train_loader:
                rr = rr.to(device, non_blocking=True)
                cpc_opt.zero_grad()
                loss, acc, _, _ = cpc(rr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cpc.parameters(), 1.0)
                cpc_opt.step()
                t_loss += loss.item()
                t_acc += acc * rr.size(0)
                t_n += rr.size(0)
            cpc_sched.step()

            cpc.eval()
            v_loss, v_acc, v_n = 0.0, 0.0, 0
            for rr, lbl, _ in cpc_val_loader:
                rr = rr.to(device, non_blocking=True)
                with torch.no_grad():
                    loss, acc, _, _ = cpc(rr)
                v_loss += loss.item()
                v_acc += acc * rr.size(0)
                v_n += rr.size(0)

            tl = t_loss / max(len(cpc_train_loader), 1)
            ta = t_acc / max(t_n, 1)
            vl = v_loss / max(len(cpc_val_loader), 1)
            va = v_acc / max(v_n, 1)

            if va > best_cpc_val_acc:
                best_cpc_val_acc = va
                cpc_patience_ctr = 0
                torch.save(cpc.state_dict(), "cpc_model_v10.pth")
            else:
                cpc_patience_ctr += 1

            run.log({"cpc_epoch": ep + 1, "cpc_train_loss": tl,
                      "cpc_train_acc": ta, "cpc_val_loss": vl, "cpc_val_acc": va})
            pbar.update(1)
            pbar.write(f"CPC {ep+1}: t_loss={tl:.4f} t_acc={ta:.2f} v_loss={vl:.4f} v_acc={va:.2f} "
                        f"[patience {cpc_patience_ctr}/{config['cpc_patience']}]")
            if cpc_patience_ctr >= config["cpc_patience"]:
                logger.info(f"CPC early stop at ep {ep+1}. Best val acc: {best_cpc_val_acc:.2f}")
                break
        pbar.close()
        cpc.load_state_dict(torch.load("cpc_model_v10.pth", weights_only=True))

    # =================================================================
    # STAGE 2: Multi-modal time-bin classification
    # =================================================================
    logger.info("=== STAGE 2: Multi-Modal Time-Bin Classification ===")

    bin_edges = config["bin_edges"]
    num_classes = len(bin_edges) - 1

    cls_train_ds = CPCTimeBinDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        bin_edges=bin_edges,
        validation_split=config["validation_split"], train=True,
    )
    cls_val_ds = CPCTimeBinDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        bin_edges=bin_edges,
        validation_split=config["validation_split"], train=False,
    )
    logger.info(f"TimeBin train: {len(cls_train_ds)}, val: {len(cls_val_ds)}")

    # Pre-compute HRV features
    hrv_train_cache = "plots/debug_signal/hrv_train_v10.npz"
    hrv_val_cache = "plots/debug_signal/hrv_val_v10.npz"

    # Reuse Phase 1 cache for training set if available
    phase1_cache = "plots/debug_signal/features_cache.npz"
    if os.path.exists(phase1_cache) and not os.path.exists(hrv_train_cache):
        logger.info(f"Copying Phase 1 HRV cache to {hrv_train_cache}")
        cached = np.load(phase1_cache)
        np.savez_compressed(hrv_train_cache, X=cached["X"])

    hrv_train = precompute_hrv(cls_train_ds.data.numpy(), hrv_train_cache)
    hrv_val = precompute_hrv(cls_val_ds.data.numpy(), hrv_val_cache)

    # Per-patient normalization (removes inter-patient baseline, isolates dynamics)
    logger.info("Recovering patient IDs and normalizing per-patient...")
    train_pids, n_train_patients = recover_patient_ids(config, processed_dataset_path, "train")
    val_pids, n_val_patients = recover_patient_ids(config, processed_dataset_path, "validation")
    logger.info(f"  Train: {n_train_patients} patients, Val: {n_val_patients} patients")

    hrv_train = normalize_per_patient(hrv_train, train_pids, n_train_patients)
    hrv_val = normalize_per_patient(hrv_val, val_pids, n_val_patients)

    hrv_train_t = torch.tensor(hrv_train, dtype=torch.float32)
    hrv_val_t = torch.tensor(hrv_val, dtype=torch.float32)

    # Build composite datasets
    train_rr = cls_train_ds.data
    train_labels = cls_train_ds.bin_labels
    val_rr = cls_val_ds.data
    val_labels = cls_val_ds.bin_labels

    train_dataset = TensorDataset(train_rr, train_labels, hrv_train_t)
    val_dataset = TensorDataset(val_rr, val_labels, hrv_val_t)

    train_bin_counts = np.bincount(train_labels.numpy(), minlength=num_classes)
    logger.info(f"Train bin dist: {dict(zip(BIN_NAMES[:num_classes], train_bin_counts.tolist()))}")

    cls_train_loader = DataLoader(
        train_dataset, batch_size=config["cls_batch_size"], shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    cls_val_loader = DataLoader(
        val_dataset, batch_size=config["cls_batch_size"], shuffle=False,
        drop_last=False, num_workers=4, pin_memory=True,
    )

    # Multi-modal classification head
    cls_head = MultiModalTimeBinHead(
        context_dim=config["context_dim"],
        latent_dim=config["latent_dim"],
        hrv_dim=HRV_FEATURE_DIM,
        num_classes=num_classes,
        dropout=config["dropout"],
    ).to(device)

    class_weights = 1.0 / train_bin_counts.clip(min=1).astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device, dtype=torch.float32),
    )

    warmup_epochs = config["cls_head_warmup_epochs"]
    backbone_lr = config["cls_lr"] * config["cls_backbone_lr_scale"]

    for p in cpc.encoder.parameters():
        p.requires_grad = False
    for p in cpc.ar_block.parameters():
        p.requires_grad = False

    cls_opt = optim.AdamW(cls_head.parameters(), lr=config["cls_lr"], weight_decay=1e-2)
    cls_sched = optim.lr_scheduler.CosineAnnealingLR(
        cls_opt, T_max=config["cls_epochs"], eta_min=config["cls_lr"] * 0.01,
    )

    cls_t_losses, cls_v_losses = [], []
    cls_t_f1s, cls_v_f1s = [], []
    cls_t_accs, cls_v_accs = [], []
    best_val_f1 = 0.0
    patience_counter = 0

    pbar = tqdm(total=config["cls_epochs"], desc="Stage 2: MultiModal TimeBin")
    for ep in range(config["cls_epochs"]):

        if ep == warmup_epochs:
            logger.info("Unfreezing backbone for fine-tuning")
            for p in cpc.encoder.parameters():
                p.requires_grad = True
            for p in cpc.ar_block.parameters():
                p.requires_grad = True
            cls_opt = optim.AdamW([
                {"params": cls_head.parameters(), "lr": config["cls_lr"]},
                {"params": cpc.encoder.parameters(), "lr": backbone_lr},
                {"params": cpc.ar_block.parameters(), "lr": backbone_lr},
            ], weight_decay=1e-2)
            cls_sched = optim.lr_scheduler.CosineAnnealingLR(
                cls_opt,
                T_max=config["cls_epochs"] - warmup_epochs,
                eta_min=backbone_lr * 0.01,
            )
            patience_counter = 0

        backbone_frozen = ep < warmup_epochs
        phase_tag = "head-only" if backbone_frozen else "fine-tune"

        cls_head.train()
        if backbone_frozen:
            cpc.eval()
        else:
            cpc.train()

        e_loss = 0.0
        t_preds, t_tgts = [], []

        for rr, tgt, hrv in cls_train_loader:
            rr = rr.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            hrv = hrv.to(device, non_blocking=True)

            cls_opt.zero_grad()
            if backbone_frozen:
                with torch.no_grad():
                    z, c = extract_features(cpc, rr)
            else:
                z, c = extract_features(cpc, rr)

            logits = cls_head(c[:, -1, :], z[:, -1, :], hrv)
            loss = criterion(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cls_head.parameters(), 1.0)
            if not backbone_frozen:
                torch.nn.utils.clip_grad_norm_(cpc.encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(cpc.ar_block.parameters(), 1.0)
            cls_opt.step()

            e_loss += loss.item()
            t_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            t_tgts.append(tgt.detach().cpu().numpy())

        cls_sched.step()

        t_preds = np.concatenate(t_preds)
        t_tgts = np.concatenate(t_tgts)
        t_acc, t_f1m, t_f1w, t_prec, t_rec, _ = compute_multiclass_metrics(t_tgts, t_preds, num_classes)
        tl = e_loss / max(len(cls_train_loader), 1)

        cls_head.eval()
        cpc.eval()
        v_loss = 0.0
        v_preds, v_tgts = [], []
        with torch.no_grad():
            for rr, tgt, hrv in cls_val_loader:
                rr = rr.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                hrv = hrv.to(device, non_blocking=True)
                z, c = extract_features(cpc, rr)
                logits = cls_head(c[:, -1, :], z[:, -1, :], hrv)
                loss = criterion(logits, tgt)
                v_loss += loss.item()
                v_preds.append(logits.argmax(dim=1).cpu().numpy())
                v_tgts.append(tgt.cpu().numpy())

        v_preds = np.concatenate(v_preds)
        v_tgts = np.concatenate(v_tgts)
        v_acc, v_f1m, v_f1w, v_prec, v_rec, cm = compute_multiclass_metrics(v_tgts, v_preds, num_classes)
        vl = v_loss / max(len(cls_val_loader), 1)

        cls_t_losses.append(tl)
        cls_v_losses.append(vl)
        cls_t_f1s.append(t_f1m)
        cls_v_f1s.append(v_f1m)
        cls_t_accs.append(t_acc)
        cls_v_accs.append(v_acc)

        cm_fig = plot_confusion_matrix(cm, BIN_NAMES[:num_classes], ep + 1)
        pca_fig = plot_per_class_acc(cm, BIN_NAMES[:num_classes], ep + 1)

        run.log({
            "cls_epoch": ep + 1, "cls_phase": phase_tag,
            "cls_lr": cls_opt.param_groups[0]["lr"],
            "cls_train_loss": tl, "cls_val_loss": vl,
            "cls_train_acc": t_acc, "cls_val_acc": v_acc,
            "cls_train_f1_macro": t_f1m, "cls_val_f1_macro": v_f1m,
            "cls_train_f1_weighted": t_f1w, "cls_val_f1_weighted": v_f1w,
            "cls_train_prec_macro": t_prec, "cls_val_prec_macro": v_prec,
            "cls_train_rec_macro": t_rec, "cls_val_rec_macro": v_rec,
            "cls_confusion_matrix": wandb.Image(cm_fig),
            "cls_per_class_acc": wandb.Image(pca_fig),
        })
        plt.close(cm_fig)
        plt.close(pca_fig)

        if v_f1m > best_val_f1:
            best_val_f1 = v_f1m
            patience_counter = 0
            torch.save(cls_head.state_dict(), "timebin_head_best_v10.pth")
            torch.save(cpc.state_dict(), "cpc_finetuned_best_v10.pth")
        else:
            patience_counter += 1

        pbar.update(1)
        pbar.write(
            f"CLS {ep+1} [{phase_tag}]: loss={tl:.4f} acc={t_acc:.4f} f1m={t_f1m:.4f} | "
            f"val_loss={vl:.4f} val_acc={v_acc:.4f} val_f1m={v_f1m:.4f} "
            f"[patience {patience_counter}/{config['cls_patience']}]"
        )

        if patience_counter >= config["cls_patience"]:
            logger.info(f"Early stopping at epoch {ep + 1}. Best val F1 macro: {best_val_f1:.4f}")
            break
    pbar.close()

    art = wandb.Artifact("timebin_multimodal_best_v10", type="model")
    art.add_file("timebin_head_best_v10.pth")
    run.log_artifact(art)

    ep_range = list(range(1, len(cls_t_f1s) + 1))
    f1_fig = plot_curves(ep_range, cls_t_f1s, cls_v_f1s, "F1 Macro",
                         "MultiModal TimeBin F1 Macro", "plots/timebin_v10_f1.png")
    run.log({"cls_f1_curves": wandb.Image(f1_fig)})
    plt.close(f1_fig)
    acc_fig = plot_curves(ep_range, cls_t_accs, cls_v_accs, "Accuracy",
                          "MultiModal TimeBin Accuracy", "plots/timebin_v10_acc.png")
    run.log({"cls_acc_curves": wandb.Image(acc_fig)})
    plt.close(acc_fig)
    loss_fig = plot_curves(ep_range, cls_t_losses, cls_v_losses, "Loss",
                           "MultiModal TimeBin Loss", "plots/timebin_v10_loss.png")
    run.log({"cls_loss_curves": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    logger.info(f"Training complete. Best val F1 macro: {best_val_f1:.4f}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    if "run" in locals():
        run.finish()
