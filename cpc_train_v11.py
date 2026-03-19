"""
CPC v11: Sequence-level time-bin classification
================================================
Instead of classifying isolated segments, this version groups K consecutive
SR segments from the same patient into a sequence.  A Transformer models
how per-segment features (CPC latent + CPC context + patient-normalised HRV)
evolve over time, then classifies the last position into Near/Mid/Far.

Pipeline:
  1. Load CPC checkpoint from v8
  2. Build CPCSequenceDataset (sliding windows of K=16 segments per patient)
  3. Pre-compute & cache per-patient-normalised HRV features
  4. Train SequenceTimeBinClassifier (Transformer + MLP head)
"""

import os
import logging
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import wandb
import h5py

from Model.CPC import CPC
from Model.SequenceClassifier import SequenceTimeBinClassifier
from Utils.Dataset.CPCSequenceDataset import CPCSequenceDataset

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

BIN_NAMES = ["Near (0-33%)", "Mid (33-67%)", "Far (67-100%)"]

# ======================================================================
# HRV Feature Computation (reused from v10)
# ======================================================================


def filter_segment(seg_flat, iqr_factor=5.0):
    q1, q3 = np.percentile(seg_flat, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-8:
        return seg_flat
    lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    return seg_flat[(seg_flat >= lo) & (seg_flat <= hi)]


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
    filtered = filter_segment(seg_flat_raw)
    artifact_rate = 1.0 - len(filtered) / max(len(seg_flat_raw), 1)
    rr = filtered if len(filtered) > 10 else seg_flat_raw
    diffs = np.diff(rr)
    return np.array([
        artifact_rate,
        np.mean(rr), np.std(rr), np.median(rr),
        np.percentile(rr, 75) - np.percentile(rr, 25),
        np.percentile(rr, 5), np.percentile(rr, 95),
        float(np.mean(((rr - np.mean(rr)) / (np.std(rr) + 1e-8)) ** 3)),
        float(np.mean(((rr - np.mean(rr)) / (np.std(rr) + 1e-8)) ** 4)),
        np.std(rr) / (np.abs(np.mean(rr)) + 1e-8),
        np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else 0.0,
        np.std(diffs) if len(diffs) > 0 else 0.0,
        np.mean(np.abs(diffs) > 0.5) if len(diffs) > 0 else 0.0,
        _dfa_alpha1(rr),
        _sample_entropy(rr, m=2, r_frac=0.2),
    ], dtype=np.float64)


HRV_DIM = 15


def precompute_hrv(data_np, cache_path):
    if os.path.exists(cache_path):
        logger.info(f"Loading cached HRV features from {cache_path}")
        return np.load(cache_path)["X"]
    logger.info(f"Computing HRV features for {len(data_np)} segments...")
    n = len(data_np)
    X = np.empty((n, HRV_DIM), dtype=np.float64)
    for i in range(n):
        X[i] = compute_hrv_features(data_np[i].flatten())
        if (i + 1) % 2000 == 0:
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
# Per-patient normalisation
# ======================================================================


def normalize_per_patient(X, patient_ids, n_patients):
    X_norm = np.copy(X)
    for pid in range(n_patients):
        mask = patient_ids == pid
        if mask.sum() < 2:
            continue
        mu = X[mask].mean(axis=0)
        sigma = X[mask].std(axis=0)
        sigma[sigma < 1e-10] = 1.0
        X_norm[mask] = (X[mask] - mu) / sigma
    X_norm[np.isnan(X_norm)] = 0.0
    return X_norm


# ======================================================================
# Feature extraction helpers
# ======================================================================


def extract_features(cpc, rr):
    """Encode RR segments through frozen CPC encoder + AR block."""
    b, t, w = rr.shape
    z_seq = cpc.encoder(rr.view(-1, w)).view(b, t, -1)
    z_seq = F.normalize(z_seq, dim=-1)
    c_seq = cpc.ar_block(z_seq)
    return z_seq, c_seq


def encode_sequence(cpc, rr_seq, hrv_seq):
    """Build per-segment feature vectors for a batch of sequences.

    Args:
        cpc: frozen CPC model
        rr_seq: (B, K, num_windows, window_size) -- K consecutive segments
        hrv_seq: (B, K, HRV_DIM) -- patient-normalised HRV per segment
    Returns:
        segment_features: (B, K, latent_dim + context_dim + HRV_DIM)
    """
    B, K, num_windows, window_size = rr_seq.shape

    rr_flat = rr_seq.reshape(B * K, num_windows, window_size)
    z, c = extract_features(cpc, rr_flat)
    z_last = z[:, -1, :].reshape(B, K, -1)
    c_last = c[:, -1, :].reshape(B, K, -1)

    return torch.cat([z_last, c_last, hrv_seq], dim=-1)


# ======================================================================
# Wrapper dataset: attach precomputed HRV to sequence indices
# ======================================================================


class SequenceWithHRV(Dataset):
    """Wraps CPCSequenceDataset, attaching pre-computed HRV features."""

    def __init__(self, seq_dataset, hrv_features):
        self.seq_dataset = seq_dataset
        self.hrv_features = torch.tensor(hrv_features, dtype=torch.float32)

    def __len__(self):
        return len(self.seq_dataset)

    def __getitem__(self, idx):
        rr_seq, label, seq_idxs = self.seq_dataset[idx]
        hrv_seq = self.hrv_features[seq_idxs]
        return rr_seq, hrv_seq, label


# ======================================================================
# Metrics & plotting (reused from v10)
# ======================================================================


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
    ax.imshow(cm, cmap="Blues")
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
        # Sequence params
        "seq_len": 16,
        "seq_stride": 4,
        # GRU + Attention params (lightweight)
        "d_model": 64,
        "seq_dropout": 0.4,
        "label_smoothing": 0.15,
        # Training -- backbone stays frozen (unfreezing causes overfitting)
        "cls_epochs": 120,
        "cls_batch_size": 64,
        "cls_lr": 1e-3,
        "cls_weight_decay": 0.05,
        "cls_patience": 25,
    }

    per_segment_dim = config["latent_dim"] + config["context_dim"] + HRV_DIM

    run = wandb.init(entity="eml-labs", project="CPC-SeqLevel-GRU-v11", config=config)

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/debug_signal", exist_ok=True)

    processed_dataset_path = "/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =================================================================
    # Load pretrained CPC encoder
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
    logger.info(f"Loading CPC checkpoint from {cpc_ckpt}")
    cpc.load_state_dict(torch.load(cpc_ckpt, weights_only=True))

    # =================================================================
    # Build sequence datasets
    # =================================================================
    logger.info("=== Building Sequence Datasets ===")

    train_seq_ds = CPCSequenceDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        bin_edges=config["bin_edges"], seq_len=config["seq_len"],
        seq_stride=config["seq_stride"],
        validation_split=config["validation_split"], train=True,
    )
    val_seq_ds = CPCSequenceDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        bin_edges=config["bin_edges"], seq_len=config["seq_len"],
        seq_stride=config["seq_stride"],
        validation_split=config["validation_split"], train=False,
    )
    logger.info(f"Sequences: train={len(train_seq_ds)}, val={len(val_seq_ds)}")
    logger.info(f"Underlying segments: train={len(train_seq_ds.all_data)}, val={len(val_seq_ds.all_data)}")

    # =================================================================
    # Pre-compute & patient-normalise HRV features
    # =================================================================
    hrv_train_cache = "plots/debug_signal/hrv_train_v11.npz"
    hrv_val_cache = "plots/debug_signal/hrv_val_v11.npz"

    phase1_cache = "plots/debug_signal/hrv_train_v10.npz"
    if os.path.exists(phase1_cache) and not os.path.exists(hrv_train_cache):
        logger.info(f"Reusing v10 HRV train cache -> {hrv_train_cache}")
        cached = np.load(phase1_cache)
        np.savez_compressed(hrv_train_cache, X=cached["X"])

    phase1_val_cache = "plots/debug_signal/hrv_val_v10.npz"
    if os.path.exists(phase1_val_cache) and not os.path.exists(hrv_val_cache):
        logger.info(f"Reusing v10 HRV val cache -> {hrv_val_cache}")
        cached = np.load(phase1_val_cache)
        np.savez_compressed(hrv_val_cache, X=cached["X"])

    hrv_train_raw = precompute_hrv(train_seq_ds.all_data.numpy(), hrv_train_cache)
    hrv_val_raw = precompute_hrv(val_seq_ds.all_data.numpy(), hrv_val_cache)

    train_pids = train_seq_ds.patient_ids
    val_pids = val_seq_ds.patient_ids
    n_train_patients = train_pids.max() + 1
    n_val_patients = val_pids.max() + 1
    logger.info(f"Patients: train={n_train_patients}, val={n_val_patients}")

    hrv_train_norm = normalize_per_patient(hrv_train_raw, train_pids, n_train_patients)
    hrv_val_norm = normalize_per_patient(hrv_val_raw, val_pids, n_val_patients)

    # Wrap datasets with HRV
    train_dataset = SequenceWithHRV(train_seq_ds, hrv_train_norm)
    val_dataset = SequenceWithHRV(val_seq_ds, hrv_val_norm)

    # Label distribution for the training sequences (last segment label)
    num_classes = train_seq_ds.num_classes
    train_labels_np = np.array([train_seq_ds.all_bin_labels[s[-1]].item() for s in train_seq_ds.sequences])
    train_bin_counts = np.bincount(train_labels_np, minlength=num_classes)
    logger.info(f"Train seq label dist: {dict(zip(BIN_NAMES[:num_classes], train_bin_counts.tolist()))}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["cls_batch_size"], shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["cls_batch_size"], shuffle=False,
        drop_last=False, num_workers=4, pin_memory=True,
    )

    # =================================================================
    # Build Sequence Classifier (lightweight GRU + attention)
    # =================================================================
    seq_model = SequenceTimeBinClassifier(
        per_segment_dim=per_segment_dim,
        num_classes=num_classes,
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        dropout=config["seq_dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in seq_model.parameters() if p.requires_grad)
    logger.info(f"Sequence classifier params: {n_params:,}")

    class_weights = 1.0 / train_bin_counts.clip(min=1).astype(np.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device, dtype=torch.float32),
        label_smoothing=config["label_smoothing"],
    )

    # Freeze CPC backbone entirely (unfreezing causes massive overfitting)
    for p in cpc.parameters():
        p.requires_grad = False
    cpc.eval()

    cls_opt = optim.AdamW(
        seq_model.parameters(), lr=config["cls_lr"],
        weight_decay=config["cls_weight_decay"],
    )
    cls_sched = optim.lr_scheduler.CosineAnnealingLR(
        cls_opt, T_max=config["cls_epochs"], eta_min=config["cls_lr"] * 0.01,
    )

    t_losses, v_losses = [], []
    t_f1s, v_f1s = [], []
    t_accs, v_accs = [], []
    best_val_f1 = 0.0
    patience_ctr = 0

    logger.info("=== Training Sequence Classifier ===")
    pbar = tqdm(total=config["cls_epochs"], desc="SeqLevel TimeBin")

    for ep in range(config["cls_epochs"]):

        seq_model.train()

        e_loss = 0.0
        all_preds, all_tgts = [], []

        for rr_seq, hrv_seq, tgt in train_loader:
            rr_seq = rr_seq.to(device, non_blocking=True)
            hrv_seq = hrv_seq.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            cls_opt.zero_grad()

            with torch.no_grad():
                seg_feats = encode_sequence(cpc, rr_seq, hrv_seq)

            logits = seq_model(seg_feats)
            loss = criterion(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq_model.parameters(), 1.0)
            cls_opt.step()

            e_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_tgts.append(tgt.detach().cpu().numpy())

        cls_sched.step()

        all_preds = np.concatenate(all_preds)
        all_tgts = np.concatenate(all_tgts)
        t_acc, t_f1m, t_f1w, t_prec, t_rec, _ = compute_multiclass_metrics(
            all_tgts, all_preds, num_classes
        )
        tl = e_loss / max(len(train_loader), 1)

        # Validation
        seq_model.eval()
        v_loss = 0.0
        vp, vt = [], []
        with torch.no_grad():
            for rr_seq, hrv_seq, tgt in val_loader:
                rr_seq = rr_seq.to(device, non_blocking=True)
                hrv_seq = hrv_seq.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                seg_feats = encode_sequence(cpc, rr_seq, hrv_seq)
                logits = seq_model(seg_feats)
                loss = criterion(logits, tgt)
                v_loss += loss.item()
                vp.append(logits.argmax(dim=1).cpu().numpy())
                vt.append(tgt.cpu().numpy())

        vp = np.concatenate(vp)
        vt = np.concatenate(vt)
        v_acc, v_f1m, v_f1w, v_prec, v_rec, cm = compute_multiclass_metrics(
            vt, vp, num_classes
        )
        vl = v_loss / max(len(val_loader), 1)

        t_losses.append(tl)
        v_losses.append(vl)
        t_f1s.append(t_f1m)
        v_f1s.append(v_f1m)
        t_accs.append(t_acc)
        v_accs.append(v_acc)

        cm_fig = plot_confusion_matrix(cm, BIN_NAMES[:num_classes], ep + 1)
        pca_fig = plot_per_class_acc(cm, BIN_NAMES[:num_classes], ep + 1)

        run.log({
            "epoch": ep + 1,
            "lr": cls_opt.param_groups[0]["lr"],
            "train_loss": tl, "val_loss": vl,
            "train_acc": t_acc, "val_acc": v_acc,
            "train_f1_macro": t_f1m, "val_f1_macro": v_f1m,
            "train_f1_weighted": t_f1w, "val_f1_weighted": v_f1w,
            "train_prec_macro": t_prec, "val_prec_macro": v_prec,
            "train_rec_macro": t_rec, "val_rec_macro": v_rec,
            "confusion_matrix": wandb.Image(cm_fig),
            "per_class_acc": wandb.Image(pca_fig),
        })
        plt.close(cm_fig)
        plt.close(pca_fig)

        if v_f1m > best_val_f1:
            best_val_f1 = v_f1m
            patience_ctr = 0
            torch.save(seq_model.state_dict(), "seq_classifier_best_v11.pth")
        else:
            patience_ctr += 1

        pbar.update(1)
        pbar.write(
            f"E{ep+1}: loss={tl:.4f} acc={t_acc:.4f} f1m={t_f1m:.4f} | "
            f"val_loss={vl:.4f} val_acc={v_acc:.4f} val_f1m={v_f1m:.4f} "
            f"[patience {patience_ctr}/{config['cls_patience']}]"
        )

        if patience_ctr >= config["cls_patience"]:
            logger.info(f"Early stopping at epoch {ep+1}. Best val F1 macro: {best_val_f1:.4f}")
            break

    pbar.close()

    # Final plots
    art = wandb.Artifact("seq_classifier_best_v11", type="model")
    art.add_file("seq_classifier_best_v11.pth")
    run.log_artifact(art)

    ep_range = list(range(1, len(t_f1s) + 1))
    f1_fig = plot_curves(ep_range, t_f1s, v_f1s, "F1 Macro",
                         "SeqLevel TimeBin F1 Macro", "plots/timebin_v11_f1.png")
    run.log({"f1_curves": wandb.Image(f1_fig)})
    plt.close(f1_fig)
    acc_fig = plot_curves(ep_range, t_accs, v_accs, "Accuracy",
                          "SeqLevel TimeBin Accuracy", "plots/timebin_v11_acc.png")
    run.log({"acc_curves": wandb.Image(acc_fig)})
    plt.close(acc_fig)
    loss_fig = plot_curves(ep_range, t_losses, v_losses, "Loss",
                           "SeqLevel TimeBin Loss", "plots/timebin_v11_loss.png")
    run.log({"loss_curves": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    logger.info(f"Training complete. Best val F1 macro: {best_val_f1:.4f}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    if "run" in locals():
        run.finish()
