"""
Probe A: patient-level holdout + seen/unseen identity detection.

Pipeline
1) Train TTE regressor on excellent patients (segment-level split, same as v17 probe setup)
2) Freeze model and extract GRU embeddings
3) Split excellent patient IDs into seen/unseen groups (patient-level holdout)
4) Train linear patient-ID classifier on seen patients only
5) Detect unseen patients using max-softmax confidence
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
    ids = sorted(int(p["pid"]) for p in q["patients"] if p["tier"] == "excellent")
    return ids


def best_threshold(scores, labels):
    thresholds = np.unique(scores)
    best_acc = 0.0
    best_t = float(thresholds[len(thresholds) // 2])
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best_t = float(t)
    return best_t, best_acc


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
        "probe_epochs": 50,
        "probe_lr": 1e-3,
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name="v17-probeA-seen-vs-unseen",
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
    pbar = tqdm(total=cfg["epochs"], desc="Regressor-A")
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
                f"[Regressor-A] epoch {ep+1:>3}/{cfg['epochs']} | "
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
                logger.info(f"[Regressor-A] early stop at epoch {ep+1} | best_val_MAE={best_mae:.2f}")
                break
    pbar.close()

    model.load_state_dict(best_state)
    model.eval()

    # Extract embeddings for all excellent segments
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

    # Patient-level holdout for probe
    pids = np.array(sorted(np.unique(pid_arr)))
    np.random.shuffle(pids)
    n_seen = int(len(pids) * cfg["probe_seen_patient_ratio"])
    seen_ids = set(pids[:n_seen].tolist())
    unseen_ids = set(pids[n_seen:].tolist())
    logger.info(
        f"Probe-A holdout: seen_patients={len(seen_ids)} | unseen_patients={len(unseen_ids)}"
    )

    seen_mask = np.isin(pid_arr, list(seen_ids))
    unseen_mask = np.isin(pid_arr, list(unseen_ids))

    x_seen = emb[seen_mask]
    y_seen_pid = pid_arr[seen_mask]
    x_unseen = emb[unseen_mask]

    # segment split for seen patients (train classifier on seen IDs only)
    idx_seen = np.random.permutation(len(x_seen))
    s_split = int(len(idx_seen) * cfg["probe_train_segment_ratio"])
    idx_seen_tr, idx_seen_va = idx_seen[:s_split], idx_seen[s_split:]

    seen_train_x = x_seen[idx_seen_tr]
    seen_train_pid = y_seen_pid[idx_seen_tr]
    seen_val_x = x_seen[idx_seen_va]
    logger.info(
        f"Probe-A segment split (seen patients only): "
        f"train={len(idx_seen_tr)} | val={len(idx_seen_va)} | unseen_eval={len(x_unseen)}"
    )

    pid2cls = {pid: i for i, pid in enumerate(sorted(seen_ids))}
    seen_train_y = np.array([pid2cls[p] for p in seen_train_pid], dtype=np.int64)

    # Normalize on probe-train
    mu = seen_train_x.mean(axis=0, keepdims=True)
    sd = seen_train_x.std(axis=0, keepdims=True) + 1e-8
    seen_train_x = (seen_train_x - mu) / sd
    seen_val_x = (seen_val_x - mu) / sd
    unseen_x = (x_unseen - mu) / sd

    probe = nn.Linear(cfg["hidden_dim"], len(seen_ids)).to(device)
    p_opt = optim.AdamW(probe.parameters(), lr=cfg["probe_lr"], weight_decay=1e-4)
    p_loss = nn.CrossEntropyLoss()

    tr_probe_loader = DataLoader(
        TensorDataset(
            torch.tensor(seen_train_x, dtype=torch.float32),
            torch.tensor(seen_train_y, dtype=torch.long),
        ),
        batch_size=256, shuffle=True,
    )

    best_conf_seen = None
    best_conf_unseen = None
    best_seen_cls_acc = 0.0
    pbar_probe = tqdm(total=cfg["probe_epochs"], desc="Probe-A")
    for ep in range(cfg["probe_epochs"]):
        probe.train()
        for x, y in tr_probe_loader:
            x, y = x.to(device), y.to(device)
            p_opt.zero_grad()
            logits = probe(x)
            loss = p_loss(logits, y)
            loss.backward()
            p_opt.step()

        probe.eval()
        with torch.no_grad():
            logits_seen = probe(torch.tensor(seen_val_x, dtype=torch.float32, device=device))
            probs_seen = torch.softmax(logits_seen, dim=-1)
            conf_seen = probs_seen.max(dim=-1).values.cpu().numpy()
            pred_seen = probs_seen.argmax(dim=-1).cpu().numpy()

            # seen val true labels
            seen_val_pid = y_seen_pid[idx_seen_va]
            seen_val_y = np.array([pid2cls[p] for p in seen_val_pid], dtype=np.int64)
            seen_cls_acc = float((pred_seen == seen_val_y).mean())

            logits_unseen = probe(torch.tensor(unseen_x, dtype=torch.float32, device=device))
            probs_unseen = torch.softmax(logits_unseen, dim=-1)
            conf_unseen = probs_unseen.max(dim=-1).values.cpu().numpy()

        # unseen detection score = -max_conf (higher means more likely unseen)
        y_det = np.concatenate([
            np.zeros_like(conf_seen, dtype=np.int64),
            np.ones_like(conf_unseen, dtype=np.int64),
        ])
        scores = np.concatenate([-conf_seen, -conf_unseen])
        auroc = float(roc_auc_score(y_det, scores))

        run.log({
            "probe/epoch": ep + 1,
            "probe/seen_cls_acc": seen_cls_acc,
            "probe/unseen_det_auroc": auroc,
            "probe/random_baseline_seen_cls": 1.0 / max(len(seen_ids), 1),
        })
        pbar_probe.update(1)
        pbar_probe.set_postfix({"seen_acc": f"{seen_cls_acc:.3f}", "auroc": f"{auroc:.3f}"})
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(
                f"[Probe-A] epoch {ep+1:>3}/{cfg['probe_epochs']} | "
                f"seen_cls_acc={seen_cls_acc:.3f} | unseen_AUROC={auroc:.3f} | "
                f"random={1.0/max(len(seen_ids),1):.3f}"
            )

        if seen_cls_acc > best_seen_cls_acc:
            best_seen_cls_acc = seen_cls_acc
            best_conf_seen = conf_seen.copy()
            best_conf_unseen = conf_unseen.copy()
    pbar_probe.close()

    # Final unseen detection stats at best seen-cls checkpoint
    y_det = np.concatenate([
        np.zeros_like(best_conf_seen, dtype=np.int64),
        np.ones_like(best_conf_unseen, dtype=np.int64),
    ])
    scores = np.concatenate([-best_conf_seen, -best_conf_unseen])
    final_auroc = float(roc_auc_score(y_det, scores))
    thr, bin_acc = best_threshold(scores, y_det)

    results = {
        "experiment": "v17_probe_A_seen_vs_unseen",
        "n_excellent_patients": len(excellent_ids),
        "n_seen_patients_for_probe": len(seen_ids),
        "n_unseen_patients_for_probe": len(unseen_ids),
        "seen_cls_random_baseline": 1.0 / max(len(seen_ids), 1),
        "best_seen_classification_acc": float(best_seen_cls_acc),
        "unseen_detection_auroc": final_auroc,
        "unseen_detection_best_threshold": float(thr),
        "unseen_detection_best_binary_acc": float(bin_acc),
        "note": "High seen-cls acc and high unseen-detection AUROC imply strong patient-specific encoding.",
    }
    out = os.path.join(root, "v17_probeA_seen_unseen_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Probe-A done | seen-cls acc={best_seen_cls_acc:.4f} | "
        f"unseen AUROC={final_auroc:.4f} | random={1.0/max(len(seen_ids),1):.4f}"
    )
    run.log({
        "final/seen_cls_acc": best_seen_cls_acc,
        "final/unseen_det_auroc": final_auroc,
        "final/unseen_det_best_binary_acc": bin_acc,
    })

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
finally:
    if "run" in locals():
        run.finish()

