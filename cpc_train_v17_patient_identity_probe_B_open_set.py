"""
Probe B: patient-level holdout + open-set nearest-centroid analysis.

Pipeline
1) Train TTE regressor on excellent patients (segment-level split, same as v17 probe setup)
2) Freeze model and extract GRU embeddings
3) Split excellent patient IDs into seen/unseen groups (patient-level holdout)
4) Build seen-patient centroids from seen-train embeddings
5) Evaluate:
   - seen closed-set ID accuracy (nearest centroid among seen classes)
   - unseen detection AUROC using nearest-centroid distance
"""

import json
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv
import wandb
from tqdm import tqdm

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TTEPredictor(nn.Module):
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
        bsz, steps, win = rr_windows.shape
        z = self.encoder(rr_windows.view(bsz * steps, win)).view(bsz, steps, -1)
        _, h = self.gru(z)
        return self.head(h.squeeze(0)).squeeze(-1)

    @torch.no_grad()
    def extract_h(self, rr_windows):
        bsz, steps, win = rr_windows.shape
        z = self.encoder(rr_windows.view(bsz * steps, win)).view(bsz, steps, -1)
        _, h = self.gru(z)
        return h.squeeze(0)


def load_excellent_ids(path):
    with open(path, "r") as f:
        q = json.load(f)
    return sorted(int(p["pid"]) for p in q["patients"] if p["tier"] == "excellent")


def pairwise_l2(a, b):
    # a: [N, D], b: [M, D] -> [N, M]
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    return np.sqrt(np.clip(aa + bb - 2.0 * ab, 0.0, None))


try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    cfg = {
        "seed": 42,
        "afib_length": 3600,
        "sr_length": int(1.5 * 3600),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        "epochs": 120,
        "batch_size": 64,
        "lr": 1e-3,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "patience": 15,
        "probe_seen_patient_ratio": 0.8,
        "probe_train_segment_ratio": 0.8,
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name="v17-probeB-open-set-centroid",
        config=cfg,
    )

    root = os.path.dirname(__file__)
    excellent_ids = load_excellent_ids(os.path.join(root, "v15c_patient_quality.json"))
    logger.info(f"Excellent patient IDs: {len(excellent_ids)}")

    ds_args = dict(
        processed_dataset_path="//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets",
        afib_length=cfg["afib_length"],
        sr_length=cfg["sr_length"],
        number_of_windows_in_segment=cfg["number_of_windows_in_segment"],
        stride=cfg["stride"],
        window_size=cfg["window_size"],
        validation_split=cfg["validation_split"],
    )
    train_ds = CPCTemporalDataset(**ds_args, train=True, sr_only=True)
    val_ds = CPCTemporalDataset(**ds_args, train=False, sr_only=True)

    all_x = torch.cat([train_ds.data, val_ds.data], dim=0)
    all_t = torch.cat([train_ds.times, val_ds.times], dim=0).float().clamp(min=0) / 60.0
    all_pid = torch.cat([train_ds.patient_ids, val_ds.patient_ids], dim=0)

    mask = np.isin(all_pid.numpy(), excellent_ids)
    all_x = all_x[mask]
    all_t = all_t[mask]
    all_pid = all_pid[mask]
    logger.info(f"Excellent-only pooled segments: {len(all_x)}")

    idx = torch.randperm(len(all_x))
    split = int(len(idx) * (1 - cfg["validation_split"]))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_loader = DataLoader(
        TensorDataset(all_x[tr_idx], all_t[tr_idx], all_pid[tr_idx]),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True,
    )
    va_loader = DataLoader(
        TensorDataset(all_x[va_idx], all_t[va_idx], all_pid[va_idx]),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictor(cfg["latent_dim"], cfg["hidden_dim"], cfg["dropout"]).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    loss_fn = nn.SmoothL1Loss()
    logger.info(
        f"Regressor split: train={len(tr_idx)} segs | val={len(va_idx)} segs | "
        f"batch_size={cfg['batch_size']} | device={device}"
    )

    best_mae = float("inf")
    best_state = None
    patience = 0
    pbar = tqdm(total=cfg["epochs"], desc="Regressor-B")
    for ep in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for rr, tte, _ in tr_loader:
            rr, tte = rr.to(device), tte.to(device)
            opt.zero_grad()
            loss = loss_fn(model(rr), tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item() * rr.size(0)
            train_n += rr.size(0)
        sched.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for rr, tte, _ in va_loader:
                vp.append(model(rr.to(device)).cpu().numpy())
                vt.append(tte.numpy())
        vp = np.concatenate(vp)
        vt = np.concatenate(vt)
        vmae = float(np.mean(np.abs(vp - vt)))
        vrho = float(np.nan_to_num(spearmanr(vp, vt)[0], nan=0.0))
        run.log({"tte/epoch": ep + 1, "tte/val_mae": vmae, "tte/val_rho": vrho})
        pbar.update(1)
        pbar.set_postfix({"vMAE": f"{vmae:.2f}", "vRho": f"{vrho:.3f}"})
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(
                f"[Regressor-B] epoch {ep+1:>3}/{cfg['epochs']} | "
                f"train_loss={train_loss_sum/max(train_n,1):.4f} | "
                f"val_MAE={vmae:.2f} | val_rho={vrho:.3f}"
            )
        if vmae < best_mae:
            best_mae = vmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                logger.info(f"[Regressor-B] early stop at epoch {ep+1} | best_val_MAE={best_mae:.2f}")
                break
    pbar.close()

    model.load_state_dict(best_state)
    model.eval()

    # Extract all embeddings
    full_loader = DataLoader(
        TensorDataset(all_x, all_t, all_pid), batch_size=cfg["batch_size"], shuffle=False
    )
    emb, pid_arr = [], []
    with torch.no_grad():
        for rr, _, pid in full_loader:
            emb.append(model.extract_h(rr.to(device)).cpu().numpy())
            pid_arr.append(pid.numpy())
    emb = np.concatenate(emb)
    pid_arr = np.concatenate(pid_arr).astype(int)

    # Patient-level holdout
    pids = np.array(sorted(np.unique(pid_arr)))
    np.random.shuffle(pids)
    n_seen = int(len(pids) * cfg["probe_seen_patient_ratio"])
    seen_ids = set(pids[:n_seen].tolist())
    unseen_ids = set(pids[n_seen:].tolist())
    logger.info(
        f"Probe-B holdout: seen_patients={len(seen_ids)} | unseen_patients={len(unseen_ids)}"
    )

    seen_mask = np.isin(pid_arr, list(seen_ids))
    unseen_mask = np.isin(pid_arr, list(unseen_ids))

    x_seen = emb[seen_mask]
    y_seen_pid = pid_arr[seen_mask]
    x_unseen = emb[unseen_mask]

    # Split seen segments into centroid-train vs seen-val
    idx_seen = np.random.permutation(len(x_seen))
    s_split = int(len(idx_seen) * cfg["probe_train_segment_ratio"])
    idx_seen_tr, idx_seen_va = idx_seen[:s_split], idx_seen[s_split:]

    x_seen_tr = x_seen[idx_seen_tr]
    y_seen_tr = y_seen_pid[idx_seen_tr]
    x_seen_va = x_seen[idx_seen_va]
    y_seen_va = y_seen_pid[idx_seen_va]
    logger.info(
        f"Probe-B segment split (seen patients only): "
        f"train={len(idx_seen_tr)} | val={len(idx_seen_va)} | unseen_eval={len(x_unseen)}"
    )

    # Normalize with seen-train stats
    mu = x_seen_tr.mean(axis=0, keepdims=True)
    sd = x_seen_tr.std(axis=0, keepdims=True) + 1e-8
    x_seen_tr = (x_seen_tr - mu) / sd
    x_seen_va = (x_seen_va - mu) / sd
    x_unseen = (x_unseen - mu) / sd

    # Build centroids by seen patient
    centroid_ids = sorted(seen_ids)
    centroids = []
    for pid in centroid_ids:
        centroids.append(x_seen_tr[y_seen_tr == pid].mean(axis=0))
    centroids = np.stack(centroids, axis=0)  # [C, D]
    pid2cls = {pid: i for i, pid in enumerate(centroid_ids)}

    # Seen closed-set accuracy
    d_seen = pairwise_l2(x_seen_va, centroids)
    pred_cls_seen = np.argmin(d_seen, axis=1)
    true_cls_seen = np.array([pid2cls[p] for p in y_seen_va], dtype=np.int64)
    seen_closed_acc = float((pred_cls_seen == true_cls_seen).mean())
    min_dist_seen = d_seen[np.arange(len(d_seen)), pred_cls_seen]

    # Unseen open-set distances
    d_unseen = pairwise_l2(x_unseen, centroids)
    min_dist_unseen = np.min(d_unseen, axis=1)

    # Unseen detection AUROC using min distance (higher => unseen)
    y_det = np.concatenate([
        np.zeros_like(min_dist_seen, dtype=np.int64),
        np.ones_like(min_dist_unseen, dtype=np.int64),
    ])
    scores = np.concatenate([min_dist_seen, min_dist_unseen])
    auroc = float(roc_auc_score(y_det, scores))

    # Threshold from seen-val 95th percentile
    thr95 = float(np.percentile(min_dist_seen, 95.0))
    pred_unseen = (scores >= thr95).astype(np.int64)
    binary_acc = float((pred_unseen == y_det).mean())
    tpr = float((pred_unseen[len(min_dist_seen):] == 1).mean())
    fpr = float((pred_unseen[:len(min_dist_seen)] == 1).mean())

    run.log({
        "probe/seen_closed_set_acc": seen_closed_acc,
        "probe/unseen_det_auroc": auroc,
        "probe/unseen_det_thr95_binary_acc": binary_acc,
        "probe/unseen_det_thr95_tpr": tpr,
        "probe/unseen_det_thr95_fpr": fpr,
    })

    results = {
        "experiment": "v17_probe_B_open_set_centroid",
        "n_excellent_patients": len(excellent_ids),
        "n_seen_patients_for_probe": len(seen_ids),
        "n_unseen_patients_for_probe": len(unseen_ids),
        "seen_closed_set_accuracy": seen_closed_acc,
        "unseen_detection_auroc": auroc,
        "threshold_95pct_seen": thr95,
        "threshold_95_binary_acc": binary_acc,
        "threshold_95_tpr_unseen": tpr,
        "threshold_95_fpr_seen": fpr,
    }
    out = os.path.join(root, "v17_probeB_open_set_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Probe-B done | seen-closed-acc={seen_closed_acc:.4f} | "
        f"unseen AUROC={auroc:.4f} | thr95 acc={binary_acc:.4f}"
    )
    logger.info(
        f"Probe-B threshold stats | thr95={thr95:.4f} | TPR_unseen={tpr:.4f} | FPR_seen={fpr:.4f}"
    )

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
finally:
    if "run" in locals():
        run.finish()

