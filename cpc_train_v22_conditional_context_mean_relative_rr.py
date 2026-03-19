"""
v22a — Conditional modeling (NOT invariant): patient context from RR support.

At inference/training time:
  - For each patient episode, we sample:
      * support_k segments (RR windows only) from that patient
      * query_k segments (RR windows + TTE labels) from that patient
  - The model encodes each support segment -> GRU hidden state h
  - Patient context = mean of support embeddings
  - For each query segment, prediction uses [h_query ; context] (concat)

Evaluation:
  - Patient-level split (train patients disjoint from val patients)
  - Within each val patient, support/query come from the SAME patient.
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


class TTEPredictorConditional(nn.Module):
    """
    Encoder -> GRU -> h
    Query head: head([h_query ; c_context])
    where context c is mean(h_support).
    """

    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_h(self, rr_windows: torch.Tensor) -> torch.Tensor:
        """
        rr_windows: [N, steps, window_size]
        returns: [N, hidden_dim] from GRU final hidden state
        """
        n, steps, w = rr_windows.shape
        z = self.encoder(rr_windows.view(n * steps, w)).view(n, steps, -1)
        _, h = self.gru(z)
        return h.squeeze(0)

    def forward(self, support_rr: torch.Tensor, query_rr: torch.Tensor) -> torch.Tensor:
        """
        support_rr: [P, support_k, steps, window_size]
        query_rr:   [P, query_k, steps, window_size]
        returns:    [P, query_k] predicted TTE (minutes)
        """
        p, s_k, steps, w = support_rr.shape
        _, q_k, _, _ = query_rr.shape

        support_flat = support_rr.view(p * s_k, steps, w)
        query_flat = query_rr.view(p * q_k, steps, w)

        h_support = self.encode_h(support_flat).view(p, s_k, -1)  # [P, support_k, D]
        context = h_support.mean(dim=1)  # [P, D]

        h_query = self.encode_h(query_flat).view(p, q_k, -1)  # [P, query_k, D]
        context_exp = context.unsqueeze(1).expand(-1, q_k, -1)  # [P, query_k, D]

        h_cat = torch.cat([h_query, context_exp], dim=-1)  # [P, query_k, 2D]
        preds = self.head(h_cat).squeeze(-1)  # [P, query_k]
        return preds


def relative_rr_remove_patient_baseline(all_data, all_pids, eps=1e-8):
    """
    all_data: Tensor [N, T, W]
    all_pids: Tensor [N]
    For each patient id p: transform x -> (x - mu_p) / (sd_p + eps)
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
    Build centroids from train patients using embedding h (not context concat),
    then compute:
      - seen closed-set accuracy
      - unseen open-set AUROC (val patients treated as unseen)
    """
    model.eval()
    # Extract embeddings for all provided segments
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
            pid = pid.cpu().numpy()
            # encode_h expects [N, steps, window_size]
            h = model.encode_h(rr).cpu().numpy()
            emb_all.append(h)
            pid_all.append(pid)
    emb_all = np.concatenate(emb_all)
    pid_all = np.concatenate(pid_all).astype(int)

    tr_mask = np.isin(pid_all, list(train_pid_set))
    va_mask = np.isin(pid_all, list(val_pid_set))
    tr_emb, tr_pid = emb_all[tr_mask], pid_all[tr_mask]
    va_emb = emb_all[va_mask]
    va_pid = pid_all[va_mask]

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

    # AUROC can fail if y is constant (should not happen)
    try:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y, s))
    except Exception:
        auroc = 0.5

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
        "validation_split": 0.15,  # fraction of patients to val
        "epochs": 120,
        "lr": 1e-3,
        "batch_patients": 8,  # P patients per episode
        "support_k": 5,
        "query_k": 3,
        "episodes_per_epoch": 150,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "patience": 20,
        "relative_rr": True,
        "relative_rr_eps": 1e-8,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name="v22a-conditional-context-mean-relative-rr",
        config=config,
    )

    # Load excellent patient IDs
    with open("v15c_patient_quality.json", "r") as f:
        quality = json.load(f)
    excellent_pids = {int(p["pid"]) for p in quality["patients"] if p["tier"] == "excellent"}
    logger.info(f"Excellent patients: {len(excellent_pids)}")

    # Load dataset (SR-only)
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

    all_data = torch.cat([train_ds_full.data, val_ds_full.data], dim=0)  # [N, steps, W]
    all_times = torch.cat([train_ds_full.times, val_ds_full.times], dim=0)
    all_pids = torch.cat([train_ds_full.patient_ids, val_ds_full.patient_ids], dim=0)

    # Filter excellent-only patients
    mask_ex = np.isin(all_pids.numpy(), list(excellent_pids))
    all_data = all_data[mask_ex]
    all_times = all_times[mask_ex]
    all_pids = all_pids[mask_ex]

    all_tte_min = all_times.float().clamp(min=0) / 60.0

    rel_stats = None
    if config["relative_rr"]:
        logger.info("Applying relative RR...")
        all_data, rel_stats = relative_rr_remove_patient_baseline(
            all_data, all_pids, eps=config["relative_rr_eps"]
        )

    # Patient-level split
    uniq = np.unique(all_pids.numpy())
    np.random.shuffle(uniq)
    split = int(len(uniq) * (1 - config["validation_split"]))
    train_pids = set(uniq[:split].tolist())
    val_pids = set(uniq[split:].tolist())
    assert not train_pids.intersection(val_pids), "Patient leakage detected!"

    # Build patient->indices mapping for train/val splits
    all_pids_np = all_pids.numpy().astype(int)
    train_indices_by_pid = {}
    val_indices_by_pid = {}
    for pid in train_pids:
        idx = np.where(all_pids_np == pid)[0]
        train_indices_by_pid[pid] = idx
    for pid in val_pids:
        idx = np.where(all_pids_np == pid)[0]
        val_indices_by_pid[pid] = idx

    logger.info(
        f"Patient split: {len(train_pids)} train | {len(val_pids)} val | "
        f"segments total={len(all_data)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictorConditional(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01
    )

    best_val_mae = float("inf")
    patience_ctr = 0
    best_state = None

    def sample_episode(split_patient_ids, indices_by_pid):
        """
        Returns:
          support_rr: [P, support_k, steps, W]
          query_rr:   [P, query_k, steps, W]
          query_tte:  [P, query_k]
          query_pids: [P, query_k] (pid repeated per query)
        """
        pids = list(split_patient_ids)
        chosen = np.random.choice(pids, size=min(config["batch_patients"], len(pids)), replace=False)
        p = len(chosen)

        support_list = []
        query_list = []
        tte_list = []
        pid_list = []

        for pid in chosen:
            idx_pool = indices_by_pid[pid]
            s_k = config["support_k"]
            q_k = config["query_k"]

            s_idx = np.random.choice(idx_pool, size=s_k, replace=len(idx_pool) < s_k)
            q_idx = np.random.choice(idx_pool, size=q_k, replace=len(idx_pool) < q_k)

            support_list.append(all_data[s_idx])
            query_list.append(all_data[q_idx])
            tte_list.append(all_tte_min[q_idx])
            pid_list.append(np.full((q_k,), pid, dtype=np.int64))

        support_rr = torch.stack(support_list, dim=0)  # [P, support_k, steps, W]
        query_rr = torch.stack(query_list, dim=0)  # [P, query_k, steps, W]
        query_tte = torch.stack(tte_list, dim=0)  # [P, query_k]
        query_pids = torch.stack([torch.tensor(p, dtype=torch.long) for p in pid_list], dim=0)

        return support_rr, query_rr, query_tte, query_pids

    for epoch in range(config["epochs"]):
        model.train()
        ep_preds = []
        ep_actual = []
        ep_pids = []
        ep_loss_sum = 0.0
        ep_n = 0

        for _ in range(config["episodes_per_epoch"]):
            support_rr, query_rr, query_tte, query_pids = sample_episode(
                train_pids, train_indices_by_pid
            )
            support_rr = support_rr.to(device)
            query_rr = query_rr.to(device)
            query_tte = query_tte.to(device)

            optimizer.zero_grad()
            pred = model(support_rr, query_rr)  # [P, query_k]
            loss = loss_fn(pred, query_tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss_sum += loss.item() * pred.numel()
            ep_n += pred.numel()

            ep_preds.append(pred.detach().cpu().numpy().reshape(-1))
            ep_actual.append(query_tte.detach().cpu().numpy().reshape(-1))
            ep_pids.append(query_pids.numpy().reshape(-1))

        scheduler.step()

        ep_preds = np.concatenate(ep_preds)
        ep_actual = np.concatenate(ep_actual)
        ep_pids = np.concatenate(ep_pids)

        train_mae = float(np.mean(np.abs(ep_preds - ep_actual)))
        train_rho = float(np.nan_to_num(spearmanr(ep_preds, ep_actual)[0], nan=0.0))

        # ── Validation (deterministic per patient: first support_k are support, rest are query) ──
        model.eval()
        v_preds = []
        v_actual = []
        v_pids_all = []
        v_loss_sum = 0.0
        v_n = 0

        with torch.no_grad():
            for pid in sorted(val_pids):
                idx_pool = val_indices_by_pid[pid]
                if len(idx_pool) < (config["support_k"] + config["query_k"]):
                    continue
                # deterministic support/query partition
                s_idx = idx_pool[: config["support_k"]]
                q_idx = idx_pool[config["support_k"] :]
                if len(q_idx) == 0:
                    continue

                # Option: to keep query budget fixed, take up to query_k random from q_idx
                # Here we use all remaining as queries for stability.
                support_rr = all_data[s_idx].unsqueeze(0).to(device)  # [1, s_k, steps, W]
                query_rr = all_data[q_idx].unsqueeze(0).to(device)  # [1, q, steps, W]
                query_tte = all_tte_min[q_idx].to(device).unsqueeze(0)  # [1, q]

                pred = model(support_rr, query_rr).squeeze(0)  # [q]
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
                "val/n_val_segments": int(len(v_preds)),
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"[v22a] ep {epoch+1:>3}/{config['epochs']} | "
                f"train MAE={train_mae:.2f} rho={train_rho:.3f} | "
                f"val MAE={val_mae:.2f} rho={val_rho:.3f} R2={val_r2:.3f} | "
                f"pp_rho_mean={v_mean_pp_rho:.3f}"
            )

        if patience_ctr >= config["patience"]:
            logger.info(f"[v22a] early stop at epoch {epoch+1} best val MAE={best_val_mae:.2f}")
            break

    # Final evaluation with best_state
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()

    # Centroid diagnostic on h only (for interpretability)
    logger.info("[v22a] running centroid diagnostic on embeddings h ...")
    all_rr_for_diag = all_data  # excellent-only already
    all_pid_for_diag = all_pids
    cd = centroid_diagnostic_h(
        model,
        all_rr_for_diag,
        all_pid_for_diag,
        train_pid_set=train_pids,
        val_pid_set=val_pids,
        device=device,
    )

    logger.info(f"[v22a] centroid probe: seen_acc={cd['seen_acc']:.4f} unseen_auroc={cd['unseen_auroc']:.4f}")
    run.log(
        {
            "final/centroid_seen_acc": cd["seen_acc"],
            "final/centroid_unseen_auroc": cd["unseen_auroc"],
        }
    )

    # Save final plot (quick scatter for best checkpoint on val queries)
    with torch.no_grad():
        v_preds, v_actual = [], []
        for pid in sorted(val_pids):
            idx_pool = val_indices_by_pid[pid]
            if len(idx_pool) <= config["support_k"]:
                continue
            s_idx = idx_pool[: config["support_k"]]
            q_idx = idx_pool[config["support_k"] :]
            if len(q_idx) == 0:
                continue
            support_rr = all_data[s_idx].unsqueeze(0).to(device)
            query_rr = all_data[q_idx].unsqueeze(0).to(device)
            query_tte = all_tte_min[q_idx].cpu().numpy().reshape(-1)
            pred = model(support_rr, query_rr).squeeze(0).cpu().numpy().reshape(-1)
            v_preds.append(pred)
            v_actual.append(query_tte)
        v_preds = np.concatenate(v_preds)
        v_actual = np.concatenate(v_actual)

    global_mae = float(np.mean(np.abs(v_preds - v_actual)))
    global_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
    ss_res = np.sum((v_actual - v_preds) ** 2)
    ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
    global_r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

    fig = plt.figure(figsize=(8, 6))
    lim = max(v_actual.max(), v_preds.max()) * 1.05
    plt.scatter(v_actual, v_preds, alpha=0.35, s=10)
    plt.plot([0, lim], [0, lim], "r--", lw=1)
    plt.xlabel("Actual TTE (min)")
    plt.ylabel("Predicted TTE (min)")
    plt.title(f"v22a Conditional context mean | MAE={global_mae:.2f}, rho={global_rho:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = "v22a_conditional_context_mean_results.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    run.log(
        {
            "final/global_mae": global_mae,
            "final/global_rho": global_rho,
            "final/global_r2": global_r2,
            "final/plot": wandb.Image(fig),
        }
    )

    logger.info(
        f"[v22a] DONE | val MAE={global_mae:.2f} rho={global_rho:.3f} R2={global_r2:.3f} | centroid_seen_acc={cd['seen_acc']:.4f} centroid_unseen_auroc={cd['unseen_auroc']:.4f}"
    )

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")

finally:
    if "run" in locals():
        run.finish()

