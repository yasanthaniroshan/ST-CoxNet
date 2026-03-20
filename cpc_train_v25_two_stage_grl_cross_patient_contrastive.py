"""
v25 — Two-stage training: GRL+Cross-patient contrastive (Stage 1) then
TTE regression fine-tune (Stage 2) on excellent patients only.

Why two stages:
- Stage 1 shapes a patient-invariant embedding space using:
    * Adversarial patient-ID invariance (GRL + patient-ID classifier CE)
    * Cross-patient contrastive structure (Gaussian soft labels on TTE)
  without forcing the embedding to regress TTE immediately.
- Stage 2 turns off adversarial + contrastive objectives and fine-tunes
  TTE regression (SmoothL1) using a smaller LR. This often reduces
  instability / representation drift.
"""

import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.autograd import Function as TorchFunction
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


class _GradReverse(TorchFunction):
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

    def set_lambda(self, val: float):
        self.lam = float(val)

    def forward(self, x):
        return _GradReverse.apply(x, self.lam)


# ═══════════════ Model ═══════════════


class TTEPredictorGRLContrastive(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        hidden_dim=128,
        dropout=0.2,
        n_patients=10,
        proj_dim=64,
    ):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)

        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.grl = GRL()
        self.patient_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_patients),
        )

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def _encode_h(self, rr_windows):
        # rr_windows: [B, steps, window_size]
        B, T, W = rr_windows.shape
        z = self.encoder(rr_windows.view(B * T, W)).view(B, T, -1)
        _, h = self.gru(z)
        return h.squeeze(0)  # [B, hidden_dim]

    def forward(self, rr_windows):
        h = self._encode_h(rr_windows)
        tte_pred = self.tte_head(h).squeeze(-1)  # [B]
        pat_logits = self.patient_head(self.grl(h))  # [B, n_patients]
        z_proj = F.normalize(self.proj_head(h), dim=-1)  # [B, proj_dim]
        return tte_pred, pat_logits, z_proj

    @torch.no_grad()
    def extract_h(self, rr_windows):
        return self._encode_h(rr_windows)


# ═══════════════ Losses ═══════════════


def cross_patient_contrastive_loss(
    z_proj,
    tte_min,
    pids,
    temperature=0.1,
    sigma_tte=10.0,
):
    """
    Soft supervised contrastive loss across patients.

    Positives: segments from different patients with similar TTE.
    weights_ij = exp(-|tte_i - tte_j|^2 / (2 sigma^2)) for pid_i != pid_j.

    InfoNCE-style: log-softmax over all targets (excluding self).
    """
    B = z_proj.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_proj.device)

    logits = (z_proj @ z_proj.T) / temperature  # [B, B]
    self_mask = torch.eye(B, dtype=torch.bool, device=z_proj.device)
    cross_patient = pids.unsqueeze(0) != pids.unsqueeze(1)  # [B, B]

    tte_diff = (tte_min.unsqueeze(0) - tte_min.unsqueeze(1)).abs()
    weights = torch.exp(-tte_diff**2 / (2.0 * sigma_tte**2))
    weights = weights * cross_patient.float()

    # Avoid in-place ops to keep autograd happy
    weights = weights.masked_fill(self_mask, 0.0)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = F.log_softmax(logits, dim=1)
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    w_sum = weights.sum(dim=1)  # [B]
    loss_per = -(weights * log_prob).sum(dim=1) / w_sum.clamp(min=1e-8)

    valid = w_sum > 1e-8
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


def centroid_diagnostic(model, train_loader, val_loader, train_pid_set, val_pid_set, device):
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
    md_va = np.min(d_va, axis=1)

    # AUROC: "seen" should be closer => smaller distances.
    # We compute both seen and unseen distances.
    md_tr = d_tr[np.arange(len(d_tr)), pred_c]
    y = np.concatenate([np.zeros(len(md_tr), dtype=int), np.ones(len(md_va), dtype=int)])
    s = np.concatenate([md_tr, md_va])
    try:
        auroc = float(roc_auc_score(y, s))
    except ValueError:
        auroc = 0.5

    return {"seen_acc": seen_acc, "unseen_auroc": auroc}


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


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
        # stage 1: invariance + contrastive
        "stage1_epochs": 80,
        "stage1_lr": 1e-3,
        "lambda_adv_max": 1.0,
        "lambda_adv_ramp_epochs": 30,
        "lambda_contrast": 0.5,
        "contrastive_temperature": 0.1,
        "contrastive_sigma_tte": 10.0,
        "stage1_patience": 20,
        # stage 2: TTE regression fine-tune
        "stage2_epochs": 120,
        "stage2_lr": 3e-4,
        "stage2_weight_decay": 1e-4,
        "stage2_patience": 25,
        # batching / model
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "proj_dim": 64,
        "P": 8,
        "K": 10,
        "batches_per_epoch": 80,
        "relative_rr": True,
    }

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name="v25-two-stage-grl-contrastive-then-regression",
        config=config,
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Load excellent patient IDs ──
    with open("v15c_patient_quality.json", "r") as f:
        quality = json.load(f)
    excellent_pids = {int(p["pid"]) for p in quality["patients"] if p["tier"] == "excellent"}
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
        all_data, _rr_stats = relative_rr_remove_patient_baseline(all_data, all_pids)

    # ── Patient-level split ──
    uniq = np.unique(all_pids.numpy())
    np.random.shuffle(uniq)
    sp = int(len(uniq) * (1 - config["validation_split"]))
    train_pid_set = set(uniq[:sp].tolist())
    val_pid_set = set(uniq[sp:].tolist())
    assert not train_pid_set.intersection(val_pid_set)

    pnp = all_pids.numpy()
    tr_m = np.isin(pnp, list(train_pid_set))
    va_m = np.isin(pnp, list(val_pid_set))

    train_data = all_data[tr_m]
    train_tte = all_tte_min[tr_m]
    train_pids_seg = all_pids[tr_m]

    val_data = all_data[va_m]
    val_tte = all_tte_min[va_m]
    val_pids_seg = all_pids[va_m]

    n_train_patients = len(train_pid_set)
    n_val_patients = len(val_pid_set)
    logger.info(
        f"Patient split: {n_train_patients} train / {n_val_patients} val | "
        f"Segments: {len(train_data)} train / {len(val_data)} val"
    )

    # Patient-ID class mapping for classifier head (train patients only)
    pid2cls = {pid: i for i, pid in enumerate(sorted(train_pid_set))}
    train_cls = torch.tensor([pid2cls[int(p)] for p in train_pids_seg.numpy()], dtype=torch.long)

    # ── DataLoaders ──
    train_torch = TensorDataset(train_data, train_tte, train_pids_seg, train_cls)
    val_torch = TensorDataset(val_data, val_tte, val_pids_seg)

    batch_sampler = PatientBatchSampler(
        train_pids_seg,
        P=config["P"],
        K=config["K"],
        batches_per_epoch=config["batches_per_epoch"],
    )
    train_loader = DataLoader(train_torch, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_torch, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Diagnostic loaders need (rr, _, pid)
    train_loader_diag = DataLoader(
        TensorDataset(train_data, train_tte, train_pids_seg),
        batch_size=512,
        shuffle=False,
    )
    val_loader_diag = DataLoader(
        TensorDataset(val_data, val_tte, val_pids_seg),
        batch_size=512,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictorGRLContrastive(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        n_patients=n_train_patients,
        proj_dim=config["proj_dim"],
    ).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | {device}")

    # ──────────────────────────────────────────────────────────────────────
    # Stage 1: train encoder+gru with invariance + contrastive
    # ──────────────────────────────────────────────────────────────────────
    set_requires_grad(model.proj_head, True)
    set_requires_grad(model.patient_head, True)
    set_requires_grad(model.tte_head, False)  # no regression loss in stage 1

    optimizer1 = optim.AdamW(
        model.parameters(),
        lr=config["stage1_lr"],
        weight_decay=1e-4,
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=config["stage1_epochs"], eta_min=config["stage1_lr"] * 0.01
    )
    pat_loss_fn = nn.CrossEntropyLoss()

    best_stage1_val = float("inf")  # lower contrastive loss is better
    patience_ctr = 0
    stage1_best_path = "v25_stage1_best.pth"

    for epoch in range(config["stage1_epochs"]):
        lam_adv = config["lambda_adv_max"] * min(
            1.0, epoch / max(config["lambda_adv_ramp_epochs"], 1)
        )
        model.grl.set_lambda(lam_adv)
        model.train()

        ep_pat_loss = 0.0
        ep_con_loss = 0.0
        ep_n = 0
        ep_pat_correct = 0
        tte_log_preds = []
        tte_log_actual = []

        for rr, tte, pids, cls in train_loader:
            rr = rr.to(device)
            tte = tte.to(device)
            pids = pids.to(device)
            cls = cls.to(device)

            optimizer1.zero_grad()
            tte_pred, pat_logits, z_proj = model(rr)

            l_pat = pat_loss_fn(pat_logits, cls)
            l_con = cross_patient_contrastive_loss(
                z_proj,
                tte,
                pids,
                temperature=config["contrastive_temperature"],
                sigma_tte=config["contrastive_sigma_tte"],
            )
            loss = lam_adv * l_pat + config["lambda_contrast"] * l_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer1.step()

            bsz = rr.size(0)
            ep_n += bsz
            ep_pat_loss += l_pat.item() * bsz
            ep_con_loss += l_con.item() * bsz
            ep_pat_correct += (pat_logits.argmax(1) == cls).sum().item()

            # purely for logging sanity
            tte_log_preds.append(tte_pred.detach().cpu().numpy())
            tte_log_actual.append(tte.detach().cpu().numpy())

        scheduler1.step()
        ep_pat_loss /= max(ep_n, 1)
        ep_con_loss /= max(ep_n, 1)
        pat_acc = ep_pat_correct / max(ep_n, 1)

        tte_log_preds = np.concatenate(tte_log_preds)
        tte_log_actual = np.concatenate(tte_log_actual)
        rho_stage1 = float(np.nan_to_num(spearmanr(tte_log_preds, tte_log_actual)[0], nan=0.0))

        # ── Stage 1 validation: compute contrastive loss on val batches ──
        model.eval()
        val_con_losses = []
        val_pat_losses = []
        with torch.no_grad():
            for rr, tte, pids in val_loader:
                rr = rr.to(device)
                tte = tte.to(device)
                pids = pids.to(device)
                tte_pred, pat_logits, z_proj = model(rr)

                # For patient classifier: val cls mapping is not available (classifier trained on train pids).
                # So we only log contrastive loss during stage 1 early stopping.
                l_con = cross_patient_contrastive_loss(
                    z_proj,
                    tte,
                    pids,
                    temperature=config["contrastive_temperature"],
                    sigma_tte=config["contrastive_sigma_tte"],
                )
                val_con_losses.append(l_con.item())

        val_con_mean = float(np.mean(val_con_losses)) if val_con_losses else float("inf")

        if val_con_mean < best_stage1_val:
            best_stage1_val = val_con_mean
            patience_ctr = 0
            torch.save(model.state_dict(), stage1_best_path)
        else:
            patience_ctr += 1

        run.log(
            {
                "stage1/epoch": epoch + 1,
                "stage1/lambda_adv": lam_adv,
                "stage1/train/pat_loss": ep_pat_loss,
                "stage1/train/contrastive_loss": ep_con_loss,
                "stage1/train/pat_acc": pat_acc,
                "stage1/train/rho_pred_vs_tte": rho_stage1,
                "stage1/val/contrastive_loss": val_con_mean,
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == config["stage1_epochs"] - 1:
            logger.info(
                f"[v25][stage1] ep {epoch+1:>3}/{config['stage1_epochs']} | "
                f"pat_loss={ep_pat_loss:.4f} con_loss={ep_con_loss:.4f} "
                f"pat_acc={pat_acc:.3f} val_con={val_con_mean:.4f} λ={lam_adv:.2f}"
            )

        if patience_ctr >= config["stage1_patience"]:
            logger.info(f"[v25][stage1] Early stop ep {epoch+1} | best val_con={best_stage1_val:.4f}")
            break

    # Load best stage1 weights
    model.load_state_dict(torch.load(stage1_best_path, weights_only=True))

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2: fine-tune TTE regression only
    # ──────────────────────────────────────────────────────────────────────
    set_requires_grad(model.proj_head, False)
    set_requires_grad(model.patient_head, False)
    set_requires_grad(model.tte_head, True)
    set_requires_grad(model.encoder, True)
    set_requires_grad(model.gru, True)
    model.grl.set_lambda(0.0)

    optimizer2 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["stage2_lr"],
        weight_decay=config["stage2_weight_decay"],
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=config["stage2_epochs"], eta_min=config["stage2_lr"] * 0.01
    )
    loss_fn = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    patience_ctr = 0
    stage2_best_path = "v25_best.pth"

    for epoch in range(config["stage2_epochs"]):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        tte_preds, tte_actual = [], []

        for rr, tte, pids, cls in train_loader:
            rr = rr.to(device)
            tte = tte.to(device)
            optimizer2.zero_grad()

            tte_pred, _pat_logits, _z_proj = model(rr)
            l = loss_fn(tte_pred, tte)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()

            bsz = rr.size(0)
            ep_loss += l.item() * bsz
            ep_n += bsz
            tte_preds.append(tte_pred.detach().cpu().numpy())
            tte_actual.append(tte.detach().cpu().numpy())

        scheduler2.step()
        ep_loss /= max(ep_n, 1)
        tte_preds = np.concatenate(tte_preds)
        tte_actual = np.concatenate(tte_actual)
        train_mae = float(np.mean(np.abs(tte_preds - tte_actual)))
        train_rho = float(np.nan_to_num(spearmanr(tte_preds, tte_actual)[0], nan=0.0))

        # validation
        model.eval()
        v_preds, v_actual = [], []
        v_pids_all = []
        with torch.no_grad():
            for rr, tte, pids in val_loader:
                rr = rr.to(device)
                tte_pred, _pat_logits, _z_proj = model(rr)
                v_preds.append(tte_pred.cpu().numpy())
                v_actual.append(tte.numpy())
                v_pids_all.append(pids.numpy())

        v_preds = np.concatenate(v_preds)
        v_actual = np.concatenate(v_actual)
        v_pids_all = np.concatenate(v_pids_all).astype(int)
        val_mae = float(np.mean(np.abs(v_preds - v_actual)))
        val_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
        ss_res = np.sum((v_actual - v_preds) ** 2)
        ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
        val_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        v_pp = compute_per_patient_rho(v_preds, v_actual, v_pids_all)
        val_pp_mean = float(np.mean(list(v_pp.values()))) if v_pp else 0.0

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_ctr = 0
            torch.save(model.state_dict(), stage2_best_path)
        else:
            patience_ctr += 1

        run.log(
            {
                "stage2/epoch": epoch + 1,
                "stage2/train/mae": train_mae,
                "stage2/train/rho": train_rho,
                "stage2/train/loss": ep_loss,
                "stage2/val/mae": val_mae,
                "stage2/val/rho": val_rho,
                "stage2/val/r2": val_r2,
                "stage2/val/pp_rho_mean": val_pp_mean,
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == config["stage2_epochs"] - 1:
            logger.info(
                f"[v25][stage2] ep {epoch+1:>3}/{config['stage2_epochs']} | "
                f"train_mae={train_mae:.2f} val_mae={val_mae:.2f} "
                f"val_rho={val_rho:.3f} pp_rho={val_pp_mean:.3f}"
            )

        if patience_ctr >= config["stage2_patience"]:
            logger.info(f"[v25][stage2] Early stop ep {epoch+1} | best val_mae={best_val_mae:.2f}")
            break

    # ── Final evaluation ──
    model.load_state_dict(torch.load(stage2_best_path, weights_only=True))
    model.eval()

    v_preds, v_actual, v_pids_all = [], [], []
    with torch.no_grad():
        for rr, tte, pids in val_loader:
            rr = rr.to(device)
            tte_pred, _pat_logits, _z_proj = model(rr)
            v_preds.append(tte_pred.cpu().numpy())
            v_actual.append(tte.numpy())
            v_pids_all.append(pids.numpy())

    v_preds = np.concatenate(v_preds)
    v_actual = np.concatenate(v_actual)
    v_pids_all = np.concatenate(v_pids_all).astype(int)

    g_mae = float(np.mean(np.abs(v_preds - v_actual)))
    g_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
    ss_res = np.sum((v_actual - v_preds) ** 2)
    ss_tot = np.sum((v_actual - v_actual.mean()) ** 2)
    g_r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))
    pp = compute_per_patient_rho(v_preds, v_actual, v_pids_all)
    g_pp = float(np.mean(list(pp.values()))) if pp else 0.0

    cd = centroid_diagnostic(
        model,
        train_loader_diag,
        val_loader_diag,
        train_pid_set,
        val_pid_set,
        device,
    )

    logger.info(f"\n{'='*65}")
    logger.info(
        f"FINAL v25 | MAE={g_mae:.2f} | rho={g_rho:.4f} | R2={g_r2:.4f} | pp_rho={g_pp:.4f}"
    )
    logger.info(
        f"centroid seen_acc={cd['seen_acc']:.4f} | centroid unseen_auroc={cd['unseen_auroc']:.4f}"
    )
    logger.info(f"{'='*65}")

    # Plot
    unique_val_pids = np.unique(v_pids_all)
    pid_colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_val_pids), 1)))
    pid_cmap = {int(pid): pid_colors[i] for i, pid in enumerate(unique_val_pids)}

    fig = plt.figure(figsize=(24, 7))
    fig.suptitle(
        f"v25 Stage1(GRL+Contrastive)->Stage2(TTE) | "
        f"MAE={g_mae:.2f} rho={g_rho:.3f} pp_rho={g_pp:.3f} R2={g_r2:.3f}",
        fontsize=12,
    )

    ax = fig.add_subplot(1, 4, 1)
    for pid in unique_val_pids:
        m = v_pids_all == int(pid)
        ax.scatter(v_actual[m], v_preds[m], alpha=0.5, s=12, c=[pid_cmap[int(pid)]], label=f"P{pid}")
    lim = max(v_actual.max(), v_preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual TTE (min)")
    ax.set_ylabel("Predicted TTE (min)")
    ax.set_title("Actual vs Predicted")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 4, 2)
    residuals = v_preds - v_actual
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", ls="--")
    ax.axvline(residuals.mean(), color="red", ls="--", label=f"mean={residuals.mean():.2f}")
    ax.set_title("Residuals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

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
    ax.set_title("Per-Patient rho")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 4, 4)
    order = np.argsort(-v_actual)
    ax.plot(v_actual[order], label="Actual", alpha=0.8, lw=1.0)
    ax.plot(v_preds[order], label="Predicted", alpha=0.8, lw=1.0)
    ax.set_xlabel("Segment (far -> close to AF)")
    ax.set_ylabel("TTE (min)")
    ax.set_title("TTE trajectory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("v25_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save results JSON (ensure native Python floats)
    results = {
        "experiment": "v25_two_stage_grl_cross_patient_contrastive_then_regression",
        "n_excellent_patients": len(excellent_pids),
        "n_train_patients": n_train_patients,
        "n_val_patients": n_val_patients,
        "n_train_segments": len(train_data),
        "n_val_segments": len(val_data),
        "relative_rr": config["relative_rr"],
        "stage1_epochs": config["stage1_epochs"],
        "stage2_epochs": config["stage2_epochs"],
        "lambda_adv_max": config["lambda_adv_max"],
        "lambda_contrast": config["lambda_contrast"],
        "contrastive_temperature": config["contrastive_temperature"],
        "contrastive_sigma_tte": config["contrastive_sigma_tte"],
        "final_global_mae": g_mae,
        "final_global_rho": g_rho,
        "final_global_r2": g_r2,
        "final_pp_rho": g_pp,
        "centroid_seen_acc": cd["seen_acc"],
        "centroid_unseen_auroc": cd["unseen_auroc"],
    }
    results = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in results.items()}

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "v25_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    run.log(
        {
            "final/mae": g_mae,
            "final/rho": g_rho,
            "final/r2": g_r2,
            "final/pp_rho": g_pp,
            "final/centroid_seen_acc": cd["seen_acc"],
            "final/centroid_unseen_auroc": cd["unseen_auroc"],
            "final_plot": wandb.Image("v25_results.png"),
        }
    )

    logger.info("Results saved. Done v25.")

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
finally:
    if "run" in locals():
        run.finish()

