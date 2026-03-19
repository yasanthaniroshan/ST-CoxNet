"""
v22c — Conditional modeling with attention-weighted patient context (relative RR).

Same as v22b (attention-weighted context from RR support segments of the same patient),
but uses relative RR preprocessing (subtract per-patient RR baseline).
"""

import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import wandb

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset


load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TTEPredictorConditionalAttention(nn.Module):
    """
    Encoder -> GRU -> h per segment
    For each query h_query, attend over support h_support to build query-specific context.
    Predict TTE from head([h_query ; context]).
    """

    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.2, temperature=0.1):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.temperature = temperature

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_h(self, rr_windows: torch.Tensor) -> torch.Tensor:
        """
        rr_windows: [N, steps, window_size]
        returns h: [N, hidden_dim]
        """
        n, steps, w = rr_windows.shape
        z = self.encoder(rr_windows.view(n * steps, w)).view(n, steps, -1)
        _, h = self.gru(z)
        return h.squeeze(0)

    def forward(self, support_rr: torch.Tensor, query_rr: torch.Tensor) -> torch.Tensor:
        """
        support_rr: [P, support_k, steps, window_size]
        query_rr:   [P, query_k, steps, window_size]
        returns preds: [P, query_k]
        """
        p, s_k, steps, w = support_rr.shape
        _, q_k, _, _ = query_rr.shape

        support_flat = support_rr.view(p * s_k, steps, w)
        query_flat = query_rr.view(p * q_k, steps, w)

        h_support = self.encode_h(support_flat).view(p, s_k, -1)  # [P,S,D]
        h_query = self.encode_h(query_flat).view(p, q_k, -1)      # [P,Q,D]

        # cosine attention
        h_support_n = h_support / (h_support.norm(dim=-1, keepdim=True) + 1e-8)
        h_query_n = h_query / (h_query.norm(dim=-1, keepdim=True) + 1e-8)

        # sim: [P,Q,S]
        sim = torch.einsum("pqd,p sd->pqs", h_query_n, h_support_n)
        weights = torch.softmax(sim / self.temperature, dim=-1)

        context = torch.einsum("pqs,p sd->pqd", weights, h_support)  # [P,Q,D]
        h_cat = torch.cat([h_query, context], dim=-1)  # [P,Q,2D]
        preds = self.head(h_cat).squeeze(-1)  # [P,Q]
        return preds


def relative_rr_remove_patient_baseline(all_data, all_pids, eps=1e-8):
    """
    all_data: Tensor [N, T, W]
    all_pids: Tensor [N]
    x -> (x - mu_p) / (sd_p + eps)
    """
    out = all_data.clone()
    pids_np = np.unique(all_pids.cpu().numpy())
    for pid in pids_np:
        mask = all_pids == int(pid)
        x = out[mask]
        mu = float(x.mean().item())
        sd = float(x.std(unbiased=False).item())
        out[mask] = (x - mu) / (sd + eps)
    return out


def compute_per_patient_rho(preds, actual, pids, min_segments=10):
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
            results[int(pid)] = float(rho)
    return results


def pairwise_l2(a, b):
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    return np.sqrt(np.clip(aa + bb - 2.0 * ab, 0.0, None))


def centroid_diagnostic_h(model, rr_data, rr_pids, train_pid_set, val_pid_set, device):
    """
    Build centroids from train patients only using embeddings h.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    emb_all = []
    pid_all = []

    with torch.no_grad():
        loader = DataLoader(
            torch.utils.data.TensorDataset(rr_data, rr_pids),
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        for rr, pid in loader:
            rr = rr.to(device)
            h = model.encode_h(rr).cpu().numpy()
            emb_all.append(h)
            pid_all.append(pid.numpy())

    emb_all = np.concatenate(emb_all)
    pid_all = np.concatenate(pid_all).astype(int)

    tr_mask = np.isin(pid_all, list(train_pid_set))
    va_mask = np.isin(pid_all, list(val_pid_set))
    tr_emb, tr_pid = emb_all[tr_mask], pid_all[tr_mask]
    va_emb = emb_all[va_mask]

    centroids_pids = sorted(train_pid_set)
    centroids = np.stack([tr_emb[tr_pid == p].mean(axis=0) for p in centroids_pids], axis=0)
    pid2c = {p: i for i, p in enumerate(centroids_pids)}

    d_tr = pairwise_l2(tr_emb, centroids)
    pred_c = np.argmin(d_tr, axis=1)
    true_c = np.array([pid2c[p] for p in tr_pid], dtype=np.int64)
    seen_acc = float((pred_c == true_c).mean())

    md_tr = d_tr[np.arange(len(d_tr)), pred_c]
    d_va = pairwise_l2(va_emb, centroids)
    md_va = np.min(d_va, axis=1)

    y = np.concatenate([np.zeros(len(md_tr), dtype=np.int64), np.ones(len(md_va), dtype=np.int64)])
    s = np.concatenate([md_tr, md_va])
    auroc = float(roc_auc_score(y, s))

    return {"seen_acc": seen_acc, "unseen_auroc": auroc}


try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    config = {
        "seed": 42,
        "afib_length": 3600,
        "sr_length": int(1.5 * 3600),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        "epochs": 120,
        "lr": 1e-3,
        "batch_patients": 8,
        "support_k": 5,
        "query_k": 3,
        "episodes_per_epoch": 150,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "patience": 20,
        "temperature": 0.1,
        "relative_rr": True,
        "relative_rr_eps": 1e-8,
    }

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name="v22c-conditional-attention-context-relative-rr",
        config=config,
    )

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Excellent patient IDs
    with open("v15c_patient_quality.json", "r") as f:
        quality = json.load(f)
    excellent_pids = {int(p["pid"]) for p in quality["patients"] if p["tier"] == "excellent"}
    logger.info(f"Excellent patients: {len(excellent_pids)}")

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

    all_data = torch.cat([train_ds_full.data, val_ds_full.data], dim=0)
    all_times = torch.cat([train_ds_full.times, val_ds_full.times], dim=0)
    all_pids = torch.cat([train_ds_full.patient_ids, val_ds_full.patient_ids], dim=0)

    mask_ex = np.isin(all_pids.numpy(), list(excellent_pids))
    all_data = all_data[mask_ex]
    all_times = all_times[mask_ex]
    all_pids = all_pids[mask_ex]

    all_tte_min = all_times.float().clamp(min=0) / 60.0

    if config["relative_rr"]:
        logger.info("Applying relative RR...")
        all_data = relative_rr_remove_patient_baseline(all_data, all_pids, eps=config["relative_rr_eps"])

    # Patient-level split
    uniq = np.unique(all_pids.numpy())
    np.random.shuffle(uniq)
    split = int(len(uniq) * (1 - config["validation_split"]))
    train_pids = set(uniq[:split].tolist())
    val_pids = set(uniq[split:].tolist())
    assert not train_pids.intersection(val_pids), "Patient leakage detected!"

    all_pids_np = all_pids.numpy().astype(int)
    train_indices_by_pid = {}
    val_indices_by_pid = {}
    for pid in train_pids:
        train_indices_by_pid[pid] = np.where(all_pids_np == pid)[0]
    for pid in val_pids:
        val_indices_by_pid[pid] = np.where(all_pids_np == pid)[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictorConditionalAttention(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    logger.info(
        f"Patient split: {len(train_pids)} train | {len(val_pids)} val | segments total={len(all_data)}"
    )
    logger.info(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | {device}")

    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01
    )

    best_val_mae = float("inf")
    patience_ctr = 0
    best_state = None

    def sample_episode(split_patient_ids, indices_by_pid):
        pids = list(split_patient_ids)
        chosen = np.random.choice(
            pids, size=min(config["batch_patients"], len(pids)), replace=False
        )
        p = len(chosen)

        support_list = []
        query_list = []
        tte_list = []

        for pid in chosen:
            idx_pool = indices_by_pid[pid]
            total_needed = config["support_k"] + config["query_k"]
            if len(idx_pool) >= total_needed:
                perm = np.random.permutation(idx_pool)[:total_needed]
                s_idx = perm[: config["support_k"]]
                q_idx = perm[config["support_k"] :]
            else:
                s_idx = np.random.choice(idx_pool, size=config["support_k"], replace=True)
                q_idx = np.random.choice(idx_pool, size=config["query_k"], replace=True)

            support_list.append(all_data[s_idx])
            query_list.append(all_data[q_idx])
            tte_list.append(all_tte_min[q_idx])

        support_rr = torch.stack(support_list, dim=0).to(device)  # [P,S,steps,W]
        query_rr = torch.stack(query_list, dim=0).to(device)      # [P,Q,steps,W]
        query_tte = torch.stack(tte_list, dim=0).to(device)      # [P,Q]
        return support_rr, query_rr, query_tte

    for epoch in range(config["epochs"]):
        model.train()
        ep_preds = []
        ep_actual = []
        ep_loss_sum = 0.0
        ep_n = 0

        for _ in range(config["episodes_per_epoch"]):
            support_rr, query_rr, query_tte = sample_episode(train_pids, train_indices_by_pid)
            optimizer.zero_grad()
            pred = model(support_rr, query_rr)
            loss = loss_fn(pred, query_tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss_sum += loss.item() * pred.numel()
            ep_n += pred.numel()
            ep_preds.append(pred.detach().cpu().numpy().reshape(-1))
            ep_actual.append(query_tte.detach().cpu().numpy().reshape(-1))

        scheduler.step()

        ep_preds = np.concatenate(ep_preds)
        ep_actual = np.concatenate(ep_actual)
        train_mae = float(np.mean(np.abs(ep_preds - ep_actual)))
        train_rho = float(np.nan_to_num(spearmanr(ep_preds, ep_actual)[0], nan=0.0))

        # Validation: deterministic support/query partition
        model.eval()
        v_preds = []
        v_actual = []
        v_pids_all = []
        v_loss_sum = 0.0
        v_n = 0

        with torch.no_grad():
            for pid in sorted(val_pids):
                idx_pool = val_indices_by_pid[pid]
                if len(idx_pool) <= config["support_k"]:
                    continue

                s_idx = idx_pool[: config["support_k"]]
                q_idx = idx_pool[config["support_k"] :]

                support_rr = all_data[s_idx].unsqueeze(0).to(device)  # [1,S,steps,W]
                query_rr = all_data[q_idx].unsqueeze(0).to(device)    # [1,Q,steps,W]
                query_tte = all_tte_min[q_idx].to(device).unsqueeze(0)  # [1,Q]

                pred = model(support_rr, query_rr).squeeze(0)

                v_loss_sum += loss_fn(pred, query_tte.squeeze(0)).item() * pred.numel()
                v_n += pred.numel()
                v_preds.append(pred.cpu().numpy().reshape(-1))
                v_actual.append(query_tte.squeeze(0).cpu().numpy().reshape(-1))
                v_pids_all.append(np.full((pred.numel(),), pid, dtype=np.int64))

        v_preds = np.concatenate(v_preds)
        v_actual = np.concatenate(v_actual)
        v_pids_all = np.concatenate(v_pids_all)

        val_mae = float(np.mean(np.abs(v_preds - v_actual)))
        val_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
        ss_res = np.sum((v_actual - v_preds) ** 2)
        ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
        val_r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

        v_per_patient = compute_per_patient_rho(v_preds, v_actual, v_pids_all)
        v_mean_pp_rho = float(np.mean(list(v_per_patient.values()))) if v_per_patient else 0.0

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        run.log(
            {
                "epoch": epoch + 1,
                "train/mae": train_mae,
                "train/rho": train_rho,
                "train/loss": ep_loss_sum / max(ep_n, 1),
                "val/mae": val_mae,
                "val/rho": val_rho,
                "val/r2": val_r2,
                "val/pp_rho_mean": v_mean_pp_rho,
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"[v22c] ep {epoch+1:>3}/{config['epochs']} | "
                f"train MAE={train_mae:.2f} rho={train_rho:.3f} | "
                f"val MAE={val_mae:.2f} rho={val_rho:.3f} R2={val_r2:.3f} | "
                f"pp_rho_mean={v_mean_pp_rho:.3f}"
            )

        if patience_ctr >= config["patience"]:
            logger.info(f"[v22c] early stop at epoch {epoch+1} best val MAE={best_val_mae:.2f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()

    logger.info("[v22c] centroid diagnostic on embeddings h ...")
    cd = centroid_diagnostic_h(
        model=model,
        rr_data=all_data,
        rr_pids=all_pids,
        train_pid_set=train_pids,
        val_pid_set=val_pids,
        device=device,
    )
    run.log({"final/centroid_seen_acc": cd["seen_acc"], "final/centroid_unseen_auroc": cd["unseen_auroc"]})

    logger.info(
        f"[v22c] DONE | centroid_seen_acc={cd['seen_acc']:.4f} centroid_unseen_auroc={cd['unseen_auroc']:.4f}"
    )

except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    raise
finally:
    if "run" in locals():
        run.finish()

