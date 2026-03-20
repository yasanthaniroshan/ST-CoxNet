"""
v24 — Adversarial Patient Invariance (GRL) + Cross-Patient Contrastive Learning.

Combined training with three objectives:
  1. TTE regression (SmoothL1Loss) — learn temporal proximity to AF onset
  2. Adversarial patient-ID classifier via Gradient Reversal Layer (GRL) —
     force embeddings to discard patient identity
  3. Cross-patient contrastive loss — pull together embeddings from
     DIFFERENT patients that share similar TTE, push apart those with
     dissimilar TTE.  Uses soft Gaussian weights on TTE distance.

Architecture:
  Encoder → GRU → h  ─┬→ TTE head        (regression)
                       ├→ GRL → PatID head (adversarial classification)
                       └→ Projection head  (contrastive, L2-normalised)

Patient-level split on excellent patients only.  Relative RR preprocessing.
PatientBatchSampler ensures each mini-batch has P patients × K segments.
"""

import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset, PatientBatchSampler

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ═══════════════ Gradient Reversal Layer ═══════════════


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def set_lambda(self, val):
        self.lam = val

    def forward(self, x):
        return _GradReverse.apply(x, self.lam)


# ═══════════════ Model ═══════════════


class TTEPredictorGRLContrastive(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.2,
                 n_patients=10, proj_dim=64):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)

        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

        self.grl = GRL()
        self.patient_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_patients),
        )

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def _encode_h(self, rr_windows):
        B, T, W = rr_windows.shape
        z = self.encoder(rr_windows.view(B * T, W)).view(B, T, -1)
        _, h = self.gru(z)
        return h.squeeze(0)  # [B, hidden_dim]

    def forward(self, rr_windows):
        h = self._encode_h(rr_windows)
        tte_pred = self.tte_head(h).squeeze(-1)     # [B]
        pat_logits = self.patient_head(self.grl(h))  # [B, n_patients]
        z_proj = F.normalize(self.proj_head(h), dim=-1)  # [B, proj_dim]
        return tte_pred, pat_logits, z_proj

    @torch.no_grad()
    def extract_h(self, rr_windows):
        return self._encode_h(rr_windows)


# ═══════════════ Losses ═══════════════


def cross_patient_contrastive_loss(z_proj, tte_min, pids,
                                   temperature=0.1, sigma_tte=10.0):
    """
    Soft supervised contrastive loss across patients.

    Positives: segments from DIFFERENT patients with similar TTE.
    Weights:   Gaussian kernel on |TTE_i - TTE_j| with bandwidth sigma_tte.
    Denominator: all non-self entries (standard InfoNCE style).

    z_proj: [B, d] L2-normalised projections
    tte_min: [B]   TTE in minutes
    pids:   [B]    patient IDs (long tensor)
    """
    B = z_proj.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_proj.device)

    logits = z_proj @ z_proj.T / temperature  # [B, B]

    self_mask = torch.eye(B, dtype=torch.bool, device=z_proj.device)
    cross_patient = pids.unsqueeze(0) != pids.unsqueeze(1)  # [B, B]

    tte_diff = (tte_min.unsqueeze(0) - tte_min.unsqueeze(1)).abs()
    weights = torch.exp(-tte_diff ** 2 / (2 * sigma_tte ** 2))
    weights = weights * cross_patient.float()
    # Avoid in-place ops: these can break autograd versioning.
    weights = weights.masked_fill(self_mask, 0.0)

    logits = logits.masked_fill(self_mask, float('-inf'))
    log_prob = F.log_softmax(logits, dim=1)
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    w_sum = weights.sum(dim=1)
    valid = w_sum > 1e-8
    loss_per = -(weights * log_prob).sum(dim=1) / w_sum.clamp(min=1e-8)

    if valid.any():
        return loss_per[valid].mean()
    return loss_per.mean()


# ═══════════════ Helpers ═══════════════


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
            results[int(pid)] = rho
    return results


@torch.no_grad()
def relative_rr_remove_patient_baseline(all_data, all_pids, eps=1e-8):
    out = all_data.clone()
    stats = {}
    for pid in np.unique(all_pids.cpu().numpy()):
        mask = all_pids == int(pid)
        x = out[mask]
        mu, sd = float(x.mean()), float(x.std(unbiased=False))
        out[mask] = (x - mu) / (sd + eps)
        stats[int(pid)] = (mu, sd)
    return out, stats


def pairwise_l2(a, b):
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.sqrt(np.clip(aa + bb - 2.0 * (a @ b.T), 0.0, None))


def centroid_diagnostic(model, train_loader, val_loader,
                        train_pid_set, val_pid_set, device):
    model.eval()
    emb_all, pid_all = [], []
    with torch.no_grad():
        for ld in [train_loader, val_loader]:
            for batch in ld:
                rr, pid = batch[0], batch[2]
                emb_all.append(model.extract_h(rr.to(device)).cpu().numpy())
                pid_all.append(pid.numpy())
    emb_all = np.concatenate(emb_all)
    pid_all = np.concatenate(pid_all).astype(int)

    tr_mask = np.isin(pid_all, list(train_pid_set))
    va_mask = np.isin(pid_all, list(val_pid_set))
    tr_emb, tr_pid = emb_all[tr_mask], pid_all[tr_mask]
    va_emb = emb_all[va_mask]

    cids = sorted(train_pid_set)
    centroids = np.stack([tr_emb[tr_pid == p].mean(0) for p in cids])
    pid2c = {p: i for i, p in enumerate(cids)}

    d_tr = pairwise_l2(tr_emb, centroids)
    pred_c = np.argmin(d_tr, axis=1)
    true_c = np.array([pid2c[p] for p in tr_pid])
    seen_acc = float((pred_c == true_c).mean())

    d_va = pairwise_l2(va_emb, centroids)
    md_tr = d_tr[np.arange(len(d_tr)), pred_c]
    md_va = np.min(d_va, axis=1)

    y = np.concatenate([np.zeros(len(md_tr)), np.ones(len(md_va))])
    s = np.concatenate([md_tr, md_va])
    try:
        auroc = float(roc_auc_score(y, s))
    except ValueError:
        auroc = 0.5
    return {"seen_acc": seen_acc, "unseen_auroc": auroc}


# ═══════════════ Main ═══════════════

try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    config = {
        "afib_length": 3600,
        "sr_length": int(1.5 * 3600),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        # training
        "epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 25,
        # model
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "proj_dim": 64,
        # batching (PatientBatchSampler)
        "P": 8,
        "K": 10,
        "batches_per_epoch": 80,
        # loss weights
        "lambda_adv_max": 1.0,
        "lambda_adv_ramp_epochs": 30,
        "lambda_contrast": 0.5,
        # contrastive hypers
        "contrastive_temperature": 0.1,
        "contrastive_sigma_tte": 10.0,
        # preprocessing
        "relative_rr": True,
    }

    run = wandb.init(
        entity="eml-labs", project="CPC-New-Temporal-Ranking",
        name="v24-grl-cross-patient-contrastive", config=config,
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Load excellent patient IDs ──
    with open("v15c_patient_quality.json", "r") as f:
        quality = json.load(f)
    excellent_pids = {int(p["pid"]) for p in quality["patients"]
                      if p["tier"] == "excellent"}
    logger.info(f"Excellent patients: {len(excellent_pids)}")

    # ── Load data ──
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

    train_ds = CPCTemporalDataset(**ds_args, train=True, sr_only=True)
    val_ds = CPCTemporalDataset(**ds_args, train=False, sr_only=True)

    all_data = torch.cat([train_ds.data, val_ds.data])
    all_times = torch.cat([train_ds.times, val_ds.times])
    all_pids = torch.cat([train_ds.patient_ids, val_ds.patient_ids])

    mask_ex = np.isin(all_pids.numpy(), list(excellent_pids))
    all_data = all_data[mask_ex]
    all_times = all_times[mask_ex]
    all_pids = all_pids[mask_ex]
    all_tte_min = all_times.float().clamp(min=0) / 60.0

    if config["relative_rr"]:
        logger.info("Applying relative RR preprocessing...")
        all_data, rr_stats = relative_rr_remove_patient_baseline(all_data, all_pids)
        logger.info(f"Relative RR: {len(rr_stats)} patients normalised")

    # ── Patient-level split ──
    uniq = np.unique(all_pids.numpy())
    np.random.shuffle(uniq)
    sp = int(len(uniq) * (1 - config["validation_split"]))
    train_pid_set = set(uniq[:sp].tolist())
    val_pid_set = set(uniq[sp:].tolist())
    assert not train_pid_set & val_pid_set

    pnp = all_pids.numpy()
    tr_m = np.isin(pnp, list(train_pid_set))
    va_m = np.isin(pnp, list(val_pid_set))

    train_data = all_data[tr_m]
    train_tte = all_tte_min[tr_m]
    train_pids_seg = all_pids[tr_m]

    val_data = all_data[va_m]
    val_tte = all_tte_min[va_m]
    val_pids_seg = all_pids[va_m]

    # Patient-ID class mapping (train patients only, for GRL head)
    pid2cls = {pid: i for i, pid in enumerate(sorted(train_pid_set))}
    n_train_patients = len(train_pid_set)
    n_val_patients = len(val_pid_set)
    train_cls = torch.tensor(
        [pid2cls[int(p)] for p in train_pids_seg.numpy()], dtype=torch.long
    )

    logger.info(
        f"Patient split: {n_train_patients} train / {n_val_patients} val | "
        f"Segments: {len(train_data)} train / {len(val_data)} val"
    )
    logger.info(f"TTE range: 0 — {float(all_tte_min.max()):.1f} min")

    # ── DataLoaders ──
    train_torch = TensorDataset(train_data, train_tte, train_pids_seg, train_cls)
    val_torch = TensorDataset(val_data, val_tte, val_pids_seg)

    batch_sampler = PatientBatchSampler(
        train_pids_seg, P=config["P"], K=config["K"],
        batches_per_epoch=config["batches_per_epoch"],
    )
    train_loader = DataLoader(
        train_torch, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_torch, batch_size=512, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictorGRLContrastive(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        n_patients=n_train_patients,
        proj_dim=config["proj_dim"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {param_count:,} | Device: {device}")

    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01,
    )
    tte_loss_fn = nn.SmoothL1Loss()
    pat_loss_fn = nn.CrossEntropyLoss()

    # ══════════════════════════════════════════════════════════════════
    #  Training loop
    # ══════════════════════════════════════════════════════════════════
    pbar = tqdm(total=config["epochs"], desc="v24 GRL+Contrastive")
    best_val_mae = float("inf")
    patience_ctr = 0

    for epoch in range(config["epochs"]):
        lam_adv = config["lambda_adv_max"] * min(
            1.0, epoch / max(config["lambda_adv_ramp_epochs"], 1)
        )
        model.grl.set_lambda(lam_adv)
        model.train()

        ep_tte_loss = 0.0
        ep_adv_loss = 0.0
        ep_con_loss = 0.0
        ep_n = 0
        ep_pat_correct = 0
        t_preds, t_actual, t_pids_ep = [], [], []

        for rr, tte, pids, cls in train_loader:
            rr = rr.to(device)
            tte = tte.to(device)
            cls = cls.to(device)
            pids_dev = pids.to(device)

            optimizer.zero_grad()
            tte_pred, pat_logits, z_proj = model(rr)

            l_tte = tte_loss_fn(tte_pred, tte)
            l_pat = pat_loss_fn(pat_logits, cls)
            l_con = cross_patient_contrastive_loss(
                z_proj, tte, pids_dev,
                temperature=config["contrastive_temperature"],
                sigma_tte=config["contrastive_sigma_tte"],
            )

            loss = l_tte + lam_adv * l_pat + config["lambda_contrast"] * l_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = rr.size(0)
            ep_tte_loss += l_tte.item() * bsz
            ep_adv_loss += l_pat.item() * bsz
            ep_con_loss += l_con.item() * bsz
            ep_n += bsz
            ep_pat_correct += (pat_logits.argmax(1) == cls).sum().item()
            t_preds.append(tte_pred.detach().cpu().numpy())
            t_actual.append(tte.cpu().numpy())
            t_pids_ep.append(pids.numpy())

        scheduler.step()

        t_preds = np.concatenate(t_preds)
        t_actual = np.concatenate(t_actual)
        t_pids_ep = np.concatenate(t_pids_ep)
        t_mae = float(np.mean(np.abs(t_preds - t_actual)))
        t_rho = float(np.nan_to_num(spearmanr(t_preds, t_actual)[0], nan=0.0))
        pat_acc = ep_pat_correct / max(ep_n, 1)

        # ── Validation ──
        model.eval()
        v_preds, v_actual, v_pids_ep = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                rr_v = batch[0].to(device)
                tte_pred_v, _, _ = model(rr_v)
                v_preds.append(tte_pred_v.cpu().numpy())
                v_actual.append(batch[1].numpy())
                v_pids_ep.append(batch[2].numpy())

        v_preds = np.concatenate(v_preds)
        v_actual = np.concatenate(v_actual)
        v_pids_ep = np.concatenate(v_pids_ep)
        v_mae = float(np.mean(np.abs(v_preds - v_actual)))
        v_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
        ss_res = np.sum((v_actual - v_preds) ** 2)
        ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
        v_r2 = 1 - ss_res / max(ss_tot, 1e-8)
        v_pp = compute_per_patient_rho(v_preds, v_actual, v_pids_ep)
        v_mean_pp = float(np.mean(list(v_pp.values()))) if v_pp else 0.0

        # Early stopping on val MAE
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            patience_ctr = 0
            torch.save(model.state_dict(), "v24_best.pth")
        else:
            patience_ctr += 1

        run.log({
            "epoch": epoch + 1,
            "lambda_adv": lam_adv,
            "train/tte_loss": ep_tte_loss / max(ep_n, 1),
            "train/adv_loss": ep_adv_loss / max(ep_n, 1),
            "train/contrastive_loss": ep_con_loss / max(ep_n, 1),
            "train/pat_acc": pat_acc,
            "train/mae": t_mae,
            "train/rho": t_rho,
            "val/mae": v_mae,
            "val/rho": v_rho,
            "val/r2": v_r2,
            "val/pp_rho": v_mean_pp,
        })

        if (epoch + 1) % 5 == 0 or epoch == config["epochs"] - 1:
            pbar.write(
                f"Ep {epoch+1:>3} | "
                f"MAE {t_mae:.2f}/{v_mae:.2f} | "
                f"ρ {t_rho:.3f}/{v_rho:.3f} | "
                f"pp_ρ {v_mean_pp:.3f} | "
                f"pat_acc {pat_acc:.3f} | "
                f"L_con {ep_con_loss/max(ep_n,1):.4f} | "
                f"λ {lam_adv:.2f}"
            )
        pbar.update(1)

        if patience_ctr >= config["patience"]:
            pbar.write(
                f"Early stop ep {epoch+1} | best MAE {best_val_mae:.2f}"
            )
            break

    pbar.close()

    # ══════════════════════════════════════════════════════════════════
    #  Final evaluation (best model)
    # ══════════════════════════════════════════════════════════════════
    model.load_state_dict(torch.load("v24_best.pth", weights_only=True))
    model.eval()

    v_preds, v_actual, v_pids_all = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            p, _, _ = model(batch[0].to(device))
            v_preds.append(p.cpu().numpy())
            v_actual.append(batch[1].numpy())
            v_pids_all.append(batch[2].numpy())

    v_preds = np.concatenate(v_preds)
    v_actual = np.concatenate(v_actual)
    v_pids_all = np.concatenate(v_pids_all)

    g_mae = float(np.mean(np.abs(v_preds - v_actual)))
    g_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
    ss_r = np.sum((v_actual - v_preds) ** 2)
    ss_t = np.sum((v_actual - v_actual.mean()) ** 2)
    g_r2 = 1 - ss_r / max(ss_t, 1e-8)
    pp = compute_per_patient_rho(v_preds, v_actual, v_pids_all)
    g_pp = float(np.mean(list(pp.values()))) if pp else 0.0

    logger.info(f"\n{'='*65}")
    logger.info(
        f"FINAL v24 | MAE={g_mae:.2f} | "
        f"ρ={g_rho:.4f} | R²={g_r2:.4f} | pp_ρ={g_pp:.4f}"
    )
    logger.info(f"{'='*65}")

    # ── Centroid diagnostic ──
    train_loader_diag = DataLoader(
        TensorDataset(train_data, train_tte, train_pids_seg),
        batch_size=512, shuffle=False,
    )
    val_loader_diag = DataLoader(
        TensorDataset(val_data, val_tte, val_pids_seg),
        batch_size=512, shuffle=False,
    )
    cd = centroid_diagnostic(
        model, train_loader_diag, val_loader_diag,
        train_pid_set, val_pid_set, device,
    )
    logger.info(
        f"Centroid probe: seen_acc={cd['seen_acc']:.4f} | "
        f"unseen_auroc={cd['unseen_auroc']:.4f}"
    )

    # ══════════════════════════════════════════════════════════════════
    #  Diagnostic plots
    # ══════════════════════════════════════════════════════════════════
    unique_val_pids = np.unique(v_pids_all)
    pid_colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_val_pids), 1)))
    pid_cmap = {pid: pid_colors[i] for i, pid in enumerate(unique_val_pids)}

    fig = plt.figure(figsize=(24, 7))
    fig.suptitle(
        f"v24 GRL+Contrastive | MAE={g_mae:.2f} | "
        f"ρ={g_rho:.3f} | pp_ρ={g_pp:.3f} | R²={g_r2:.3f} | "
        f"centroid seen={cd['seen_acc']:.2f} unseen_auroc={cd['unseen_auroc']:.2f}",
        fontsize=12,
    )

    # 1 — Actual vs Predicted
    ax = fig.add_subplot(1, 4, 1)
    for pid in unique_val_pids:
        m = v_pids_all == pid
        ax.scatter(v_actual[m], v_preds[m], alpha=0.5, s=12,
                   c=[pid_cmap[pid]], label=f"P{int(pid)}")
    lim = max(v_actual.max(), v_preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual TTE (min)"); ax.set_ylabel("Predicted TTE (min)")
    ax.set_title("Actual vs Predicted"); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # 2 — Residuals
    ax = fig.add_subplot(1, 4, 2)
    residuals = v_preds - v_actual
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", ls="--")
    ax.axvline(residuals.mean(), color="red", ls="--",
               label=f"mean={residuals.mean():.2f}")
    ax.set_title("Residuals"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3 — Per-patient ρ
    ax = fig.add_subplot(1, 4, 3)
    if pp:
        sp = sorted(pp.keys(), key=lambda p: pp[p])
        rho_vals = [pp[p] for p in sp]
        bar_colors = ["steelblue" if r >= 0 else "coral" for r in rho_vals]
        ax.barh(range(len(sp)), rho_vals, color=bar_colors, height=0.7)
        ax.set_yticks(range(len(sp)))
        ax.set_yticklabels([f"P{p}" for p in sp], fontsize=7)
        ax.axvline(g_pp, color="k", ls="--", lw=1, label=f"mean={g_pp:.3f}")
        ax.legend(fontsize=8)
    ax.set_title("Per-Patient ρ"); ax.grid(alpha=0.3)

    # 4 — TTE trajectory (sorted)
    ax = fig.add_subplot(1, 4, 4)
    order = np.argsort(-v_actual)
    ax.plot(v_actual[order], label="Actual", alpha=0.8, lw=1.0)
    ax.plot(v_preds[order], label="Predicted", alpha=0.8, lw=1.0)
    ax.set_xlabel("Segment (far → close to AF)")
    ax.set_ylabel("TTE (min)"); ax.set_title("TTE trajectory")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("v24_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to v24_results.png")

    # ── Save results ──
    results = {
        "experiment": "v24_grl_cross_patient_contrastive",
        "n_excellent_patients": len(excellent_pids),
        "n_train_patients": n_train_patients,
        "n_val_patients": n_val_patients,
        "n_train_segments": len(train_data),
        "n_val_segments": len(val_data),
        "relative_rr": config["relative_rr"],
        "lambda_adv_max": config["lambda_adv_max"],
        "lambda_contrast": config["lambda_contrast"],
        "contrastive_temperature": config["contrastive_temperature"],
        "contrastive_sigma_tte": config["contrastive_sigma_tte"],
        "best_val_mae": float(best_val_mae),
        "final_global_mae": g_mae,
        "final_global_rho": g_rho,
        "final_global_r2": g_r2,
        "final_pp_rho": g_pp,
        "centroid_seen_acc": cd["seen_acc"],
        "centroid_unseen_auroc": cd["unseen_auroc"],
    }

    # Convert numpy scalars (e.g. float32) to native Python for json serialization.
    results = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in results.items()}

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "v24_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    run.log({
        "final/mae": g_mae, "final/rho": g_rho, "final/r2": g_r2,
        "final/pp_rho": g_pp,
        "final/centroid_seen_acc": cd["seen_acc"],
        "final/centroid_unseen_auroc": cd["unseen_auroc"],
        "final_plot": wandb.Image("v24_results.png"),
    })

    logger.info("Results saved. Done v24.")

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
    if "pbar" in locals():
        pbar.close()
finally:
    if "run" in locals():
        run.finish()
