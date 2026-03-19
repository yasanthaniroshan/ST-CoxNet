"""
v23 — Pairwise ranking instead of regression.

Goal:
  For two segments x_i, x_j from the SAME patient, predict which one is closer
  to AF onset (smaller TTE is positive).

Training objective (pairwise):
  Let s_i = f(x_i) be a scalar "closeness score".
  Use logits = s_i - s_j.
  Target y_ij = 1 if TTE_i < TTE_j else 0.
  Loss = BCEWithLogitsLoss(logits, y_ij) over all valid ordered pairs within
  each patient mini-group.

Batching:
  Use PatientBatchSampler to sample P patients × K segments per patient,
  ensuring we can form pairs within each patient group.

RR preprocessing:
  relative_rr=True: subtract per-patient RR baseline (mean/std) as in v20/v22.

Evaluation:
  - Spearman rho between score and -TTE (because higher score should mean
    smaller TTE => larger -TTE)
  - Per-patient Spearman rho
  - Pairwise accuracy on val pairs
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset, PatientBatchSampler

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ScoreModel(nn.Module):
    """Encoder -> GRU -> MLP head producing a scalar score."""

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
        # rr_windows: [B, steps, window_size]
        bsz, steps, win = rr_windows.shape
        z = self.encoder(rr_windows.view(bsz * steps, win)).view(bsz, steps, -1)
        _, h = self.gru(z)
        h = h.squeeze(0)  # [B, hidden_dim]
        score = self.head(h).squeeze(-1)  # [B]
        return score


def relative_rr_remove_patient_baseline(all_data, all_pids, eps=1e-8):
    """
    all_data: [N, steps, window_size]
    all_pids: [N]
    For each patient p, transform all segments from p with:
      (x - mu_p) / (sd_p + eps)
    """
    out = all_data.clone()
    pids_np = np.unique(all_pids.cpu().numpy())
    stats = {}
    for pid in pids_np:
        mask = all_pids == int(pid)
        x = out[mask]
        mu = float(x.mean().item())
        sd = float(x.std(unbiased=False).item())
        out[mask] = (x - mu) / (sd + eps)
        stats[int(pid)] = (mu, sd)
    return out, stats


def compute_per_patient_spearman(pred_scores, tte_min, pids, min_segments=10):
    """
    Spearman between score and -tte_min (score higher => smaller tte).
    Returns dict: pid -> rho
    """
    results = {}
    for pid in np.unique(pids):
        m = pids == pid
        if m.sum() < min_segments:
            continue
        s = pred_scores[m]
        tt = tte_min[m]
        if (tt.max() - tt.min()) < 1e-6:
            continue
        rho, _ = spearmanr(s, -tt)
        if not np.isnan(rho):
            results[int(pid)] = float(rho)
    return results


def pairwise_ranking_loss(scores, tte_min, tie_eps=1e-6):
    """
    scores: [P, K]
    tte_min: [P, K]
    Builds all ordered pairs (i,j) within each patient group p.
    Returns (loss, pairwise_acc) where acc measures correct ordering.
    """
    # scores_i - scores_j
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [P, K, K]

    # y_ij = 1 if tte_i < tte_j
    t_i = tte_min.unsqueeze(2)
    t_j = tte_min.unsqueeze(1)
    y = (t_i < t_j).float()  # [P, K, K]

    # Exclude diagonal and near-ties
    diag = torch.eye(scores.shape[1], device=scores.device, dtype=torch.bool).unsqueeze(0)  # [1,K,K]
    abs_diff = (t_i - t_j).abs()
    valid = (~diag) & (abs_diff > tie_eps)

    # BCEWithLogits on logits=diff
    bce = nn.BCEWithLogitsLoss(reduction="none")
    loss_all = bce(diff, y)  # [P,K,K]
    loss = loss_all[valid].mean() if valid.any() else loss_all.mean()

    # Pairwise accuracy: pred = diff>0 => score_i > score_j => i closer => tte_i < tte_j
    pred = (diff > 0).float()
    acc = ((pred == y) & valid).float().sum() / max(valid.float().sum(), torch.tensor(1.0, device=scores.device))
    return loss, acc


def pairwise_accuracy_on_val(scores_all, tte_all, pids_all, max_pairs_per_patient=200):
    """
    Sample ordered pairs within each val patient and compute accuracy.
    Returns float acc.
    """
    accs = []
    for pid in np.unique(pids_all):
        m = pids_all == pid
        if m.sum() < 2:
            continue
        s = scores_all[m]
        t = tte_all[m]
        n = len(s)
        # sample pairs
        rng = np.random.default_rng(42 + int(pid))
        num_pairs = min(max_pairs_per_patient, n * (n - 1) // 2)
        i = rng.integers(0, n, size=num_pairs)
        j = rng.integers(0, n, size=num_pairs)
        mask = i != j
        i = i[mask]
        j = j[mask]
        if len(i) == 0:
            continue
        pred = s[i] > s[j]
        y = t[i] < t[j]
        accs.append(float((pred == y).mean()))
    return float(np.mean(accs)) if accs else 0.0


try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    @dataclass
    class Config:
        # dataset
        afib_length: int = 3600
        sr_length: int = int(1.5 * 3600)
        number_of_windows_in_segment: int = 10
        stride: int = 100
        window_size: int = 100
        validation_split: float = 0.15
        # pool
        tier: str = "excellent"
        # model
        latent_dim: int = 64
        hidden_dim: int = 128
        dropout: float = 0.2
        # train
        epochs: int = 120
        lr: float = 1e-3
        weight_decay: float = 1e-4
        patience: int = 20
        # batch sampling for pairs
        P: int = 8  # patients per batch
        K: int = 10  # segments per patient per batch (forms K^2 pairs)
        batches_per_epoch: int = 80
        tie_eps: float = 1e-6
        # rr preprocessing
        relative_rr: bool = True
        relative_rr_eps: float = 1e-8

    cfg = Config()

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name="v23-pairwise-ranking-relative-rr",
        config=cfg.__dict__,
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # Load excellent patient IDs
    with open("v15c_patient_quality.json", "r") as f:
        quality = json.load(f)
    pool_pids = {int(p["pid"]) for p in quality["patients"] if p["tier"] == cfg.tier}
    logger.info(f"Using tier={cfg.tier} patients: {len(pool_pids)}")

    processed = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    ds_args = dict(
        processed_dataset_path=processed,
        afib_length=cfg.afib_length,
        sr_length=cfg.sr_length,
        number_of_windows_in_segment=cfg.number_of_windows_in_segment,
        stride=cfg.stride,
        window_size=cfg.window_size,
        validation_split=cfg.validation_split,
    )

    train_ds_full = CPCTemporalDataset(**ds_args, train=True, sr_only=True)
    val_ds_full = CPCTemporalDataset(**ds_args, train=False, sr_only=True)

    all_data = torch.cat([train_ds_full.data, val_ds_full.data], dim=0)
    all_times = torch.cat([train_ds_full.times, val_ds_full.times], dim=0)
    all_pids = torch.cat([train_ds_full.patient_ids, val_ds_full.patient_ids], dim=0)

    mask_pool = np.isin(all_pids.cpu().numpy(), list(pool_pids))
    all_data = all_data[mask_pool]
    all_times = all_times[mask_pool]
    all_pids = all_pids[mask_pool]

    all_tte_min = all_times.float().clamp(min=0) / 60.0

    rel_stats = None
    if cfg.relative_rr:
        logger.info("Applying relative RR preprocessing...")
        all_data, rel_stats = relative_rr_remove_patient_baseline(all_data, all_pids, eps=cfg.relative_rr_eps)
        logger.info(f"Relative RR computed for {len(rel_stats)} patients")

    # Patient-level split
    uniq = np.unique(all_pids.cpu().numpy())
    np.random.shuffle(uniq)
    split = int(len(uniq) * (1 - cfg.validation_split))
    train_pid_set = set(uniq[:split].tolist())
    val_pid_set = set(uniq[split:].tolist())
    assert not train_pid_set.intersection(val_pid_set)

    pnp = all_pids.cpu().numpy()
    tr_mask = np.isin(pnp, list(train_pid_set))
    va_mask = np.isin(pnp, list(val_pid_set))

    train_data = all_data[tr_mask]
    train_tte = all_tte_min[tr_mask]
    train_pids = all_pids[tr_mask]

    val_data = all_data[va_mask]
    val_tte = all_tte_min[va_mask]
    val_pids = all_pids[va_mask]

    logger.info(
        f"Patient split: train_patients={len(train_pid_set)} val_patients={len(val_pid_set)} | "
        f"train_segments={len(train_data)} val_segments={len(val_data)}"
    )

    train_torch = TensorDataset(train_data, train_tte, train_pids)
    val_torch = TensorDataset(val_data, val_tte, val_pids)

    # Batches ensure within-patient pairing
    batch_sampler = PatientBatchSampler(train_pids, P=cfg.P, K=cfg.K, batches_per_epoch=cfg.batches_per_epoch)
    train_loader = DataLoader(
        train_torch,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(val_torch, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScoreModel(cfg.latent_dim, cfg.hidden_dim, cfg.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_metric = -1e9
    best_state = None
    patience_ctr = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        tr_accs = []

        for rr, tte, pids in train_loader:
            # rr: [P*K, steps, window]
            rr = rr.to(device)
            tte = tte.to(device)

            # reshape into [P,K]
            scores = model(rr).view(cfg.P, cfg.K)
            tte_group = tte.view(cfg.P, cfg.K)

            loss, pair_acc = pairwise_ranking_loss(scores, tte_group, tie_eps=cfg.tie_eps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_losses.append(loss.item())
            tr_accs.append(float(pair_acc.item()))

        scheduler.step()

        # Validation: score->Spearman(score,-TTE) + pairwise acc
        model.eval()
        scores_list = []
        tte_list = []
        pids_list = []
        with torch.no_grad():
            for rr, tte, pids in val_loader:
                rr = rr.to(device)
                sc = model(rr).cpu().numpy()
                scores_list.append(sc)
                tte_list.append(tte.numpy())
                pids_list.append(pids.numpy())

        scores_all = np.concatenate(scores_list)
        tte_all = np.concatenate(tte_list)
        pids_all = np.concatenate(pids_list).astype(int)

        rho, _ = spearmanr(scores_all, -tte_all)
        rho = float(np.nan_to_num(rho, nan=0.0))
        pp_rhos = compute_per_patient_spearman(scores_all, tte_all, pids_all, min_segments=10)
        mean_pp_rho = float(np.mean(list(pp_rhos.values()))) if pp_rhos else 0.0

        pair_acc_val = pairwise_accuracy_on_val(
            scores_all=scores_all,
            tte_all=tte_all,
            pids_all=pids_all,
            max_pairs_per_patient=200,
        )

        # Choose best by val rho
        if rho > best_metric:
            best_metric = rho
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(os.path.dirname(__file__), "v23_best.pth"))
        else:
            patience_ctr += 1

        run.log(
            {
                "epoch": epoch + 1,
                "train/pairwise_loss": float(np.mean(tr_losses)) if tr_losses else 0.0,
                "train/pairwise_acc": float(np.mean(tr_accs)) if tr_accs else 0.0,
                "val/spearman_score_vs_neg_tte": rho,
                "val/pp_rho_mean": mean_pp_rho,
                "val/pairwise_acc": pair_acc_val,
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            logger.info(
                f"[v23] ep {epoch+1:>3}/{cfg.epochs} | "
                f"train loss={np.mean(tr_losses):.4f} train pair_acc={np.mean(tr_accs):.3f} | "
                f"val rho={rho:.3f} val pair_acc={pair_acc_val:.3f} pp_rho_mean={mean_pp_rho:.3f}"
            )

        if patience_ctr >= cfg.patience:
            logger.info(f"[v23] Early stopping at epoch {epoch+1} | best val rho={best_metric:.3f}")
            break

    # Finalize with best state
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    logger.info(f"[v23] DONE | best val rho={best_metric:.3f}")

    # Save small results summary
    results = {
        "experiment": "v23_pairwise_ranking_relative_rr",
        "tier": cfg.tier,
        "train_patients": len(train_pid_set),
        "val_patients": len(val_pid_set),
        "P": cfg.P,
        "K": cfg.K,
        "relative_rr": cfg.relative_rr,
        "best_val_spearman_score_vs_neg_tte": best_metric,
        "pair_sampling_within_patient": True,
        "pair_def": "positive if TTE_i < TTE_j",
    }
    out_path = os.path.join(os.path.dirname(__file__), "results", "v23_pairwise_ranking_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    run.finish()

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
    raise

