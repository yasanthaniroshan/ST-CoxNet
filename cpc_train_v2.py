from Model.CPC import CPC
from Model.PredictionHead.TimePredictor import TimePredictor
from Model.CoxHead.Base import CoxHead
from Loss.DeepSurvLoss import DeepSurvLoss
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

def compute_baseline_hazard(risk, time, event):
    order = torch.argsort(time)
    time = time[order]
    event = event[order]
    risk = risk[order]

    unique_times = torch.unique(time[event == 1])

    hazards = []
    for t in unique_times:
        d = ((time == t) & (event == 1)).sum()
        risk_set = torch.exp(risk[time >= t]).sum()
        hazards.append(d / risk_set)

    hazards = torch.stack(hazards)
    cumhaz = torch.cumsum(hazards, dim=0)

    return unique_times, cumhaz

def predict_median_survival(risk, times, baseline_cumhaz):
    surv = torch.exp(-baseline_cumhaz * torch.exp(risk))

    idx = torch.where(surv <= 0.5)[0]
    if len(idx) == 0:
        return times[-1]

    return times[idx[0]]

def concordance_index(pred, target):
    """
    pred, target: [B, 1] normalized time-to-event
    Penalizes pairs where the predicted ordering disagrees with actual ordering.
    """
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    
    # All pairwise differences
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)      # [B, B]
    target_diff = target.unsqueeze(0) - target.unsqueeze(1) # [B, B]
    
    # Mask: only consider pairs where target ordering is clear (not ties)
    mask = (target_diff.abs() > 1e-4)
    
    # Concordant if signs agree
    concordant = ((pred_diff * target_diff) > 0).float()
    
    # C-index as a loss (maximize concordance = minimize 1 - concordance)
    c_index = concordant[mask].mean()
    return c_index

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
        "cox_epochs": 20,
        "dropout": 0.2,
        "temperature": 0.2,
        "latent_dim": 64,
        "context_dim": 128,
        "number_of_prediction_steps": 6,
        "batch_size": 512,
        "cpc_lr": 1e-3,
        "cox_lr": 1e-3
    }
    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-v2", 
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
    cox_epochs = config["cox_epochs"]
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
    cpc.train()
    total_loader_size = len(train_data_loader)
    pbar = tqdm(total=cpc_epochs, desc="Training CPC Model")

    for cpc_epoch in range(cpc_epochs):
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

    cox_train_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True
    )

    logger.info(f"Loaded {len(cox_train_dataset)} training segments.")

    cox_validation_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=False
    )

    logger.info(f"Loaded {len(cox_validation_dataset)} validation segments.")

    cox_train_data_loader = DataLoader(
        cox_train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    cox_validation_data_loader = DataLoader(
        cox_validation_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )



    cox = CoxHead(
        context_dim=config['context_dim'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout']
    ).to(device)

    cox_optimizer = optim.AdamW(cox.parameters(), lr=config["cox_lr"],weight_decay=1e-2)
    loss_fn = DeepSurvLoss()

    cpc.eval()

    for param in cpc.parameters():
        param.requires_grad = False

    pbar = tqdm(total=cox_epochs, desc="Training Cox Model")
    for cox_epoch in range(cox_epochs):
        total_loss = 0
        validation_loss = 0
        train_risk = []
        train_time = []
        train_event = []  
        for rr,label,time,event in cox_train_data_loader:

            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            event = event.to(device, non_blocking=True)

            cox_optimizer.zero_grad()

            with torch.no_grad():
                _,_, embeddings, context = cpc(rr)

            logits = cox(context[:, -1, :],embeddings[:,-1,:])
            loss = loss_fn(logits, time, event)

            total_loss += loss.item()
            loss.backward()
            cox_optimizer.step()

            train_risk.append(logits.detach().cpu())
            train_time.append(time.cpu())
            train_event.append(event.cpu())

        actual_times = []
        predicted_times = []

        train_event = torch.cat(train_event)
        train_time = torch.cat(train_time)
        train_risk = torch.cat(train_risk)

        times,baseline_cumhuz = compute_baseline_hazard(
            train_risk,
            train_time,
            train_event
        )

        val_risk = []
        val_time = []
        val_event = []
          
        for rr,label,time,event in cox_validation_data_loader:

            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            event = event.to(device, non_blocking=True)

            actual_times.append(time.cpu().numpy())

            with torch.no_grad():
                _,_, embeddings, context = cpc(rr)
                logits = cox(context[:, -1, :],embeddings[:,-1,:])

                loss = loss_fn(logits, time, event)
                validation_loss += loss.item()

                val_risk.append(logits.detach().cpu())
                val_time.append(time.cpu())
                val_event.append(event.cpu())

                local_pred_times = []

                for r in logits.cpu():
                    predicted_time = predict_median_survival(r, times, baseline_cumhuz)
                    local_pred_times.append(predicted_time.cpu().item())
                predicted_times.append(local_pred_times)

        val_time = torch.cat(val_time)
        val_event = torch.cat(val_event)
        val_risk = torch.cat(val_risk)
        predictions = torch.tensor([item for sublist in predicted_times for item in sublist])
        concordance = concordance_index(predictions.unsqueeze(1), val_time.unsqueeze(1))

        np_actual_times = np.concatenate(actual_times, axis=0)
        np_predicted_times = np.concatenate(predicted_times, axis=0)
        
        fig =plt.figure(figsize=(12,12))
        fig.suptitle(f"Cox Model - Epoch {cox_epoch+1}", fontsize=16)
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.scatter(np_actual_times, np_predicted_times, alpha=0.5)
        ax1.plot([0, np.max(np_actual_times)], [0, np.max(np_actual_times)], 'r--')  # Line for perfect predictions
        ax1.set_xlabel("Actual Time to Event (hours)")
        ax1.set_ylabel("Predicted Time to Event (hours)")
        ax1.set_title(f"Actual vs Predicted Time to Event - Epoch {cox_epoch+1}")
        ax1.set_xlim(0, np.max(np_actual_times)*1.1)
        ax1.set_ylim(0, np.max(np_actual_times)*1.1)
        ax1.grid()
        # Hex Bin density Plot
        ax2 = fig.add_subplot(gs[0, 1])
        hb = ax2.hexbin(np_actual_times, np_predicted_times, gridsize=50, cmap='Blues', mincnt=1)
        ax2.set_xlabel("Actual Time to Event (hours)")
        ax2.set_ylabel("Predicted Time to Event (hours)")
        ax2.set_title(f"Prediction Density - Epoch {cox_epoch+1}")
       

        # Risk Distribution Plot
        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(val_risk.cpu().numpy(), bins=50, color='blue', alpha=0.7)
        ax3.set_xlabel("Predicted Risk Score")
        ax3.set_title(f"Predicted Risk Score Distribution - Epoch {cox_epoch+1}")
        ax3.grid()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        run.log({
            "cox_epoch": cox_epoch+1,
            "cox_training_loss": total_loss / len(cox_train_data_loader),
            "cox_validation_loss": validation_loss / len(cox_validation_data_loader),
            "concordance_index": concordance.item(),
            "cox_scatter_plot": wandb.Image(fig)
        })
        plt.close(fig)
        pbar.update(1)
        pbar.set_description(f"Epoch {cox_epoch+1}")
        pbar.write(f"Epoch {cox_epoch+1}, Cox Model Loss: {total_loss / len(cox_train_data_loader):.4f}, Validation Loss: {validation_loss / len(cox_validation_data_loader):.4f} Concordance Index: {concordance.item():.4f}")

    pbar.close()
    torch.save(cox.state_dict(), "cox_model.pth")
    artifact = wandb.Artifact("cox_model", type="model")
    artifact.add_file("cox_model.pth")
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