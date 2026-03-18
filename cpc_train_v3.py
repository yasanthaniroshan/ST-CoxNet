from Model.CPC import CPC
from Model.Weibul import WeibullHead
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Utils.Dataset.CPCDataset import CPCDataset
from Utils.Dataset.CPCCoxDataset import CPCCoxDataset
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def weibull_nll(lambda_, k, t, event):
    """
    Numerically stable negative log-likelihood for Weibull.
    lambda_: (B, 1), strictly > 0
    k:       (B, 1), strictly > 0
    t:       (B,)   -> time (>= 0)
    event:   (B,)   -> 1 if event occurred, 0 if censored
    """
    eps = 1e-8

    # Shapes to (B,1)
    t = t.unsqueeze(1)
    event = event.unsqueeze(1)

    # Clamp time to avoid log(0) and extreme exponents
    t_clamped = torch.clamp(t, min=eps)

    log_t = torch.log(t_clamped)
    log_lambda = torch.log(lambda_ + eps)

    # Stable exponent term: (t / lambda_) ** k = exp(k * (log_t - log_lambda))
    log_ratio = log_t - log_lambda
    exp_arg = k * log_ratio
    exp_arg = torch.clamp(exp_arg, max=50.0)  # avoid overflow
    pow_term = torch.exp(exp_arg)

    # log pdf and log survival
    log_f = torch.log(k + eps) - log_lambda + (k - 1.0) * log_t - pow_term
    log_S = -pow_term

    log_likelihood = event * log_f + (1.0 - event) * log_S
    return -log_likelihood.mean()

def predict_median_survival(lambda_, k, t):
    """
    Median of Weibull: t_med = lambda * (ln 2)^(1 / k)
    Returns tensor of shape (B,1).
    """
    eps = 1e-8
    ln2 = torch.log(torch.tensor(2.0, device=lambda_.device, dtype=lambda_.dtype))
    inv_k = 1.0 / torch.clamp(k, min=eps)
    factor = torch.exp(inv_k * torch.log(ln2 + eps))
    t_med = lambda_ * factor
    return t_med

try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    config={
        "afib_length": 60*60,
        "sr_length": int(1.5*60*60),
        "number_of_windows_in_segment": 10,
        "stride": 20,
        "window_size": 100,
        "validation_split": 0.15,
        "cpc_epochs": 20,
        "weibull_epochs": 20,
        "dropout": 0.2,
        "temperature": 0.2,
        "latent_dim": 64,
        "context_dim": 128,
        "number_of_prediction_steps": 6,
        "batch_size": 512,
        "cpc_lr": 1e-3,
        "weibull_lr": 1e-3
    }
    run = wandb.init(
        entity="eml-labs",
        project="CPC-with-Weibull-Head", 
        config=config,
    )
        

    torch.manual_seed(42)
    np.random.seed(42)

    afib_length = config["afib_length"]
    sr_length = config["sr_length"]
    number_of_windows_in_segment = config["number_of_windows_in_segment"]
    stride = config["stride"]
    window_size = config["window_size"]
    validation_split = config["validation_split"]
    cpc_epochs = config["cpc_epochs"]
    weibull_epochs = config["weibull_epochs"]
    processed_dataset_path = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"

    train_dataset = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True
    )

    logger.info(f"Loaded {len(train_dataset)} training segments.")

    validation_dataset = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=False
    )

    logger.info(f"Loaded {len(validation_dataset)} validation segments.")

    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    validation_data_loader = DataLoader(
        validation_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cpc = CPC(
        latent_dim=config["latent_dim"],
        context_dim=config["context_dim"],
        number_of_prediction_steps=config["number_of_prediction_steps"],
        dropout=config["dropout"],
        temperature=config["temperature"]
    ).to(device)

    cpc_optimizer = optim.AdamW(cpc.parameters(), lr=config["cpc_lr"], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(cpc_optimizer, T_max=cpc_epochs, eta_min=config["cpc_lr"]*0.1)
    total_loader_size = len(train_data_loader)
    pbar = tqdm(total=cpc_epochs, desc="Training CPC Model")

    for cpc_epoch in range(cpc_epochs):
        cpc.train()
        total_samples = 0
        total_loss = 0
        total_accuracy = 0
        validation_loss = 0
        validation_accuracy = 0
        for rr,label,_ in train_data_loader:
            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            cpc_optimizer.zero_grad()
            loss, accuracy, _,_ = cpc(rr)
            total_loss += loss.item()
            total_accuracy += accuracy*rr.size(0)
            total_samples += rr.size(0)
            loss.backward()
            cpc_optimizer.step()
        scheduler.step()
        embeddings_list = []
        context_list = []
        label_list = []
        total_validation_samples = 0
        cpc.eval()
        for rr,label,_ in validation_data_loader:
            rr = rr.to(device, non_blocking=True)
            label_list.extend(label.cpu().numpy().tolist())
            with torch.no_grad():
                loss, accuracy, embeddings, context = cpc(rr)
                validation_loss += loss.item()
                validation_accuracy += accuracy*rr.size(0)
                total_validation_samples += rr.size(0)
                embeddings_list.append(embeddings.cpu().numpy())
                context_list.append(context.cpu().numpy())

        pca_emb = PCA(n_components=2,random_state=42)
        pca_ctx = PCA(n_components=2,random_state=42)

        embeddings_all = np.concatenate(embeddings_list, axis=0)
        context_all = np.concatenate(context_list, axis=0)
        # For visualization, we take the last time step's embedding and context for each segment, which represents the most recent information before the event (AFIB start). This allows us to see how well the model is capturing the differences between SR, Mixed, and AFIB segments in the latent space. If we took all time steps, it would be harder to visualize and interpret the results, as each segment would contribute multiple points to the PCA plot.
        embeddings_all = embeddings_all[:, -1, :]
        context_all = context_all[:, -1, :]
        embeddings_pca = pca_emb.fit_transform(embeddings_all)
        context_pca = pca_ctx.fit_transform(context_all)
        # -1 - SR, 0 - Mixed, 1 - AFIB
        color_map = {-1: 'blue', 0: 'green', 1: 'red'}
        colors = [color_map[l] for l in label_list]
        fig = plt.figure(figsize=(12,12))
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        # ax3 = fig.add_subplot(gs[2:, 0])
        # ax4 = fig.add_subplot(gs[:, 0])

        scatter1 = ax1.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=colors, alpha=0.7,s=10)
        ax1.set_title("PCA of Latent Embeddings")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")

        scatter2 = ax2.scatter(context_pca[:,0], context_pca[:,1], c=colors, alpha=0.7,s=10)
        ax2.set_title("PCA of Context Vectors")
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")


        # Add color mapping legend
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=color_map[k], label=lbl) for k, lbl in zip([-1, 0, 1], ['SR', 'Mixed', 'AFIB'])]
        fig.legend(handles=legend_handles, loc='upper right', title='Color Mapping')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pbar.update(1)
        pbar.set_description(f"Epoch {cpc_epoch+1}")
        pbar.write(f"Epoch {cpc_epoch+1}, Loss: {total_loss / len(train_data_loader):.4f}, Accuracy: {(total_accuracy / total_samples):.4f}, Validation Loss: {validation_loss / len(validation_data_loader):.4f}, Validation Accuracy: {(validation_accuracy / total_validation_samples):.4f}")
        run.log({
            "cpc_epoch": cpc_epoch+1,
            "cpc_lr": cpc_optimizer.param_groups[0]['lr'],
            "cpc_train_loss": total_loss / len(train_data_loader),
            "cpc_train_accuracy": (total_accuracy / total_samples) if total_samples > 0 else 0,
            "cpc_validation_loss": validation_loss / len(validation_data_loader),
            "cpc_validation_accuracy": (validation_accuracy / total_validation_samples) if total_validation_samples > 0 else 0,
            "pca": wandb.Image(fig)
        })
        plt.close(fig)

    pbar.close()
    torch.save(cpc.state_dict(), "cpc_model.pth")
    artifact = wandb.Artifact("cpc_model", type="model")
    artifact.add_file("cpc_model.pth")
    run.log_artifact(artifact)

    weibull_train_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True
    )

    logger.info(f"Loaded {len(weibull_train_dataset)} training segments.")

    t_all, e_all = [], []
    for _, _, t, e in weibull_train_dataset:
        t_all.append(t.item()); e_all.append(e.item())
    t_all = np.array(t_all)
    logger.info(f"Time range:  [{t_all.min():.4f}, {t_all.max():.4f}]")
    logger.info(f"Time mean:   {t_all.mean():.4f}  std: {t_all.std():.4f}")
    logger.info(f"Event rate:  {np.mean(e_all):.2%}")

    weibull_validation_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=False
    )

    logger.info(f"Loaded {len(weibull_validation_dataset)} validation segments.")

    weibull_train_data_loader = DataLoader(
        weibull_train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    weibull_validation_data_loader = DataLoader(
        weibull_validation_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    weibull_head = WeibullHead(
        context_dim=config['context_dim'],
        latent_dim=config['latent_dim']
    ).to(device)
    weibull_head_optimizer = optim.AdamW(weibull_head.parameters(), lr=config["weibull_lr"], weight_decay=1e-2)
    for param in cpc.parameters():
        param.requires_grad = False

    pbar = tqdm(total=weibull_epochs, desc="Training Weibull Head")
    for weibull_epoch in range(weibull_epochs):
        total_loss = 0
        validation_loss = 0
        train_time = []
        weibull_head.train()
        cpc.eval()
        for rr,label,time,event in weibull_train_data_loader:

            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            event = event.to(device, non_blocking=True)


            weibull_head_optimizer.zero_grad()

            # with torch.no_grad():
            _,_, embeddings, context = cpc(rr)

            lambda_, k_ = weibull_head(context[:, -1, :],embeddings[:,-1,:])
            loss = weibull_nll(lambda_, k_, time, event)
            total_loss += loss.item()
            loss.backward()
            weibull_head_optimizer.step()

            train_time.append(time.cpu())

        actual_times = []
        predicted_times = []


        val_time = []

        weibull_head.eval()
        cpc.eval()
        for rr,label,time,event in weibull_validation_data_loader:
            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            event = event.to(device, non_blocking=True)

            actual_times.append(time.cpu().numpy())

            with torch.no_grad():
                _,_, embeddings, context = cpc(rr)
                lamda_, k_ = weibull_head(context[:, -1, :],embeddings[:,-1,:])
                loss = weibull_nll(lamda_, k_, time, event)
                validation_loss += loss.item()

                predicted_times.append(predict_median_survival(lamda_, k_, time).cpu().numpy())

        np_actual_times = np.concatenate(actual_times, axis=0)
        np_predicted_times = np.concatenate(predicted_times, axis=0)
        
        fig = plt.figure(figsize=(12,12))
        fig.suptitle(f"Weibull Head - Epoch {weibull_epoch+1}", fontsize=16)
        gs = fig.add_gridspec(1,2)
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.scatter(np_actual_times, np_predicted_times, alpha=0.5)
        ax1.plot([0, np.max(np_actual_times)], [0, np.max(np_actual_times)], 'r--')
        ax1.set_xlabel("Actual Time to Event (hours)")
        ax1.set_ylabel("Predicted Time to Event (hours)")
        ax1.set_title(f"Actual vs Predicted Time to Event - Epoch {weibull_epoch+1}")
        ax1.set_xlim(0, np.max(np_actual_times)*1.1)
        ax1.set_ylim(0, np.max(np_actual_times)*1.1)
        ax1.grid()
        # Hex Bin density Plot
        ax2 = fig.add_subplot(gs[0, 1])
        hb = ax2.hexbin(np_actual_times, np_predicted_times, gridsize=50, cmap='Blues', mincnt=1)
        ax2.set_xlabel("Actual Time to Event (hours)")
        ax2.set_ylabel("Predicted Time to Event (hours)")
        ax2.set_title(f"Prediction Density - Epoch {weibull_epoch+1}")
       

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        run.log({
            "weibull_epoch": weibull_epoch+1,
            "weibull_training_loss": total_loss / len(weibull_train_data_loader),
            "weibull_validation_loss": validation_loss / len(weibull_validation_data_loader),
            "weibull_scatter_plot": wandb.Image(fig)
        })
        plt.close(fig)
        pbar.update(1)
        pbar.set_description(f"Epoch {weibull_epoch+1}")
        pbar.write(f"Epoch {weibull_epoch+1}, Weibull Head Loss: {total_loss / len(weibull_train_data_loader):.4f}, Validation Loss: {validation_loss / len(weibull_validation_data_loader):.4f}")

    pbar.close()
    torch.save(weibull_head.state_dict(), "weibull_head.pth")
    artifact = wandb.Artifact("weibull_head", type="model")
    artifact.add_file("weibull_head.pth")
    run.log_artifact(artifact)

except Exception as e:
    if 'logger' in locals():
        logger.error(f"An error occurred: {e}", exc_info=True)
    else:        
        print(f"An error occurred: {e}")
    if 'pbar' in locals():
        pbar.close()
finally:
    if 'run' in locals():
        run.finish()