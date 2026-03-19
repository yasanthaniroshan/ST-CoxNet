import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    classification_report,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import wandb

from Model.CPC import CPC
from Model.TimeBinHead import TimeBinHead
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


def extract_features(cpc: CPC, rr: torch.Tensor):
    b, t, w = rr.shape
    z_seq = cpc.encoder(rr.view(-1, w)).view(b, t, -1)
    z_seq = F.normalize(z_seq, dim=-1)
    c_seq = cpc.ar_block(z_seq)
    return z_seq, c_seq


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    acc = float((y_pred == y_true).mean()) if len(y_true) > 0 else 0.0
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return acc, f1_macro, f1_weighted, prec_macro, rec_macro, cm


def plot_cpc_pca(embeddings, contexts, labels, epoch):
    pca_e = PCA(n_components=2, random_state=42)
    pca_c = PCA(n_components=2, random_state=42)
    e2d = pca_e.fit_transform(embeddings)
    c2d = pca_c.fit_transform(contexts)

    cmap = {-1: "blue", 0: "green", 1: "red"}
    names = {-1: "SR", 0: "Mixed", 1: "AFIB"}
    colors = [cmap[int(l)] for l in labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.scatter(e2d[:, 0], e2d[:, 1], c=colors, alpha=0.6, s=8)
    ax1.set_title(f"Latent Embeddings PCA (epoch {epoch})")
    ax2.scatter(c2d[:, 0], c2d[:, 1], c=colors, alpha=0.6, s=8)
    ax2.set_title(f"Context PCA (epoch {epoch})")
    handles = [mpatches.Patch(color=cmap[k], label=names[k]) for k in [-1, 0, 1]]
    fig.legend(handles=handles, loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


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
        # --- Time bins ---
        "bin_edges": [0.0, 1/3, 2/3, 1.01],
        # --- Stage 1: CPC (cross-batch InfoNCE) ---
        "cpc_epochs": 50,
        "cpc_batch_size": 512,
        "cpc_lr": 1e-3,
        "cpc_patience": 8,
        # --- Stage 2: Time-bin classification (2-phase) ---
        "cls_epochs": 40,
        "cls_batch_size": 256,
        "cls_lr": 3e-4,
        "cls_backbone_lr_scale": 0.01,
        "cls_head_warmup_epochs": 8,
        "cls_patience": 10,
    }

    run = wandb.init(entity="eml-labs", project="CPC-TimeBin-Classification-v9", config=config)

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("plots", exist_ok=True)

    processed_dataset_path = "/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================================
    # STAGE 1: CPC — load v8 checkpoint or train from scratch
    # =====================================================================
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
        logger.info("=== STAGE 1: CPC Pretraining (no checkpoint found) ===")

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
                torch.save(cpc.state_dict(), "cpc_model_v9.pth")
            else:
                cpc_patience_ctr += 1

            run.log({
                "cpc_epoch": ep + 1, "cpc_train_loss": tl,
                "cpc_train_acc": ta, "cpc_val_loss": vl, "cpc_val_acc": va,
            })

            pbar.update(1)
            pbar.write(
                f"CPC {ep+1}: train_loss={tl:.4f} train_acc={ta:.2f} "
                f"val_loss={vl:.4f} val_acc={va:.2f} "
                f"[patience {cpc_patience_ctr}/{config['cpc_patience']}]"
            )

            if cpc_patience_ctr >= config["cpc_patience"]:
                logger.info(f"CPC early stop at epoch {ep+1}. Best val acc: {best_cpc_val_acc:.2f}")
                break
        pbar.close()

        cpc.load_state_dict(torch.load("cpc_model_v9.pth", weights_only=True))

    # =====================================================================
    # STAGE 2: Time-bin classification (SR segments only)
    #   Phase A: head-only warmup    Phase B: fine-tune backbone
    # =====================================================================
    logger.info("=== STAGE 2: Time-Bin Classification ===")

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

    train_bin_counts = np.bincount(cls_train_ds.bin_labels.numpy(), minlength=num_classes)
    logger.info(f"Train bin distribution: {dict(zip(BIN_NAMES[:num_classes], train_bin_counts.tolist()))}")

    cls_train_loader = DataLoader(
        cls_train_ds, batch_size=config["cls_batch_size"], shuffle=True,
        drop_last=True, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    cls_val_loader = DataLoader(
        cls_val_ds, batch_size=config["cls_batch_size"], shuffle=False,
        drop_last=False, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    cls_head = TimeBinHead(
        context_dim=config["context_dim"],
        latent_dim=config["latent_dim"],
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

    pbar = tqdm(total=config["cls_epochs"], desc="Stage 2: TimeBin Classification")
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

        for rr, bin_label, _ in cls_train_loader:
            rr = rr.to(device, non_blocking=True)
            tgt = bin_label.to(device, non_blocking=True)

            cls_opt.zero_grad()
            if backbone_frozen:
                with torch.no_grad():
                    z, c = extract_features(cpc, rr)
            else:
                z, c = extract_features(cpc, rr)

            logits = cls_head(c[:, -1, :], z[:, -1, :])
            loss = criterion(logits, tgt)
            loss.backward()
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(cls_head.parameters(), max_norm)
            if not backbone_frozen:
                torch.nn.utils.clip_grad_norm_(cpc.encoder.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(cpc.ar_block.parameters(), max_norm)
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
            for rr, bin_label, _ in cls_val_loader:
                rr = rr.to(device, non_blocking=True)
                tgt = bin_label.to(device, non_blocking=True)
                z, c = extract_features(cpc, rr)
                logits = cls_head(c[:, -1, :], z[:, -1, :])
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
            torch.save(cls_head.state_dict(), "timebin_head_best_v9.pth")
            torch.save(cpc.state_dict(), "cpc_finetuned_best_v9.pth")
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

    art = wandb.Artifact("timebin_head_best_v9", type="model")
    art.add_file("timebin_head_best_v9.pth")
    run.log_artifact(art)

    ep_range = list(range(1, len(cls_t_f1s) + 1))
    f1_fig = plot_curves(ep_range, cls_t_f1s, cls_v_f1s, "F1 Macro", "TimeBin F1 Macro", "plots/timebin_v9_f1.png")
    run.log({"cls_f1_curves": wandb.Image(f1_fig)})
    plt.close(f1_fig)
    acc_fig = plot_curves(ep_range, cls_t_accs, cls_v_accs, "Accuracy", "TimeBin Accuracy", "plots/timebin_v9_acc.png")
    run.log({"cls_acc_curves": wandb.Image(acc_fig)})
    plt.close(acc_fig)
    loss_fig = plot_curves(ep_range, cls_t_losses, cls_v_losses, "Loss", "TimeBin Loss", "plots/timebin_v9_loss.png")
    run.log({"cls_loss_curves": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    logger.info(f"Training complete. Best val F1 macro: {best_val_f1:.4f}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    if "run" in locals():
        run.finish()
