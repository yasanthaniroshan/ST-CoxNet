import logging

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader, random_split
from typing import Tuple,Dict
from torch.utils.data import Dataset
import numpy as np
import os
import random
import wandb
from Utils.Dataset.IRIDADataset import RRSequenceDataset
from Utils.Dataset.CoxDataset.RRSequenceCoxDataset import RRSequenceCoxDataset
from Model.DeepSurv import DeepSurvCox
from Model.Encoder.ResNetEncoder import Encoder
from Model.AutoregressiveBlock import ARBlock
from Model.PredictionHead.HRVPredictor.MultiStepPredictor import MultiStepHRVPredictor
from logging import getLogger, FileHandler, Formatter, INFO
import time
from tqdm import tqdm
from Loss.DeepSurvLoss import DeepSurvLoss
from Metric.CIndex import CIndex
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dotenv import load_dotenv

class CPCPreModel(nn.Module):
    def __init__(self, num_targets:int,latent_dim:int, context_dim:int,number_of_heads:int):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.context = ARBlock(latent_dim=latent_dim, context_dim=context_dim)
        self.predictor = MultiStepHRVPredictor(context_dim=context_dim, num_heads=number_of_heads,num_targets=num_targets)

    def forward(self, rr_windows: torch.Tensor) -> torch.Tensor:
        """
        rr_windows: [B, T, W] 
        Returns:
            c_seq: [B, T, context_dim]
        """
        
        B, T, W = rr_windows.shape
        z_list = []

        for t in range(T):
            z_t = self.encoder(rr_windows[:, t, :])  # [B, latent_dim]
            z_list.append(z_t)

        z_seq = torch.stack(z_list, dim=1)  # [B, T, latent_dim]
        c_seq = self.context(z_seq)         # [B, T, context_dim]

        return c_seq

def get_loss_weights(epoch, total_epochs):
    """
    Gradually shifts focus from loss_1 (Horizon 4) to loss_4 (Horizon 16).
    """
    # Progressive factor from 0 to 1
    alpha = epoch / total_epochs 
    
    # Near horizon starts strong (1.0) and decays
    w1 = 1.2
    # Mid horizon stays relatively stable
    w2 = 1.0 
    # Far horizon starts low (0.2) and becomes the priority (1.5)
    w4 = 0.8
    
    return w1, w2, w4

def training_step(
    model, 
    rr_windows: torch.Tensor, 
    hrv_targets: torch.Tensor,
    epoch,
    total_epochs:int
) -> torch.Tensor:
    """
    rr_windows: [B, T, W]  -> RR windows
    hrv_targets : [B, T, num_metrics] -> HRV targets for different horizons
    Returns:
        loss: scalar tensor
    """
    w1, w2, w4 = get_loss_weights(epoch,total_epochs)
    # Get context embeddings from model
    c_seq = model(rr_windows)  # [B, T, context_dim]
    last_context = c_seq[:, -1, :]  # [B, context_dim]
    y_pred_1, y_pred_2, y_pred_4 = model.predictor(last_context)  # Each: [B, num_metrics]
    y_true_1, y_true_2, y_true_4 = hrv_targets[:, 0, :], hrv_targets[:, 1, :], hrv_targets[:, 2, :]  # Each: [B, num_metrics]
    loss_1 = F.mse_loss(y_pred_1, y_true_1)
    loss_2 = F.mse_loss(y_pred_2, y_true_2)
    loss_4 = F.mse_loss(y_pred_4, y_true_4)
    total_loss = (w1 * loss_1) + (w2 * loss_2) + (w4 * loss_4)
    return total_loss,loss_1,loss_2,loss_4

def validation_step(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    val_loss_1 = 0.0
    val_loss_2 = 0.0
    val_loss_4 = 0.0
    count = 0

    with torch.no_grad():
        for rr_windows, hrv_targets in dataloader:
            rr_windows = rr_windows.to(device)  # [B, 1, W] if single-channel
            # Move targets to device
            hrv_targets = hrv_targets.to(device)

            # Compute loss
            loss,loss_1,loss_2,loss_4 = training_step(model, rr_windows, hrv_targets,epoch,CPC_EPOCHS)
            val_loss += loss.item()
            val_loss_1 += loss_1.item()
            val_loss_2 += loss_2.item()
            val_loss_4 += loss_4.item()

            count += 1

    return (val_loss / count,val_loss_1 / count,val_loss_2 / count,val_loss_4 / count) if count > 0 else (0.0,0.0,0.0,0.0)

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


logger = getLogger(__name__)
log_folder = os.path.join(os.getcwd(), 'Logs')
os.makedirs(log_folder, exist_ok=True)
log_path_file = os.path.join(log_folder, f'training_log_{time.strftime("%Y-%m-%d_%H-%M-%S")}.txt')
file_handler = logging.FileHandler(log_path_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


random.seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

load_dotenv()


# Hyperparameters
CPC_LR = 5e-4
COX_LR = 1e-3
CPC_EPOCHS = 50
COX_EPOCHS = 100
LATENT_SIZE = 32
CONTEXT_SIZE = 128
NUMBER_OF_TARGETS_FOR_PREDICTION = 6
NUMBER_OF_HEADS = 3
WINDOW_SIZE = 200
STRIDE = 20
HORIZONS = [2,4,8]  # Predicting HRV differences at 2, 4, and 8 windows into the future
SEQUENCE_LENGTH = 20
VALIDATION_RATIO = 0.3
BATCH_SIZE = 512
PREDICTION = "hrv difference"
NOTES = "Testing new Encoder and Prediction head architectures with adjusted learning rates and loss weighting strategy on IRIDA-AF dataset."


EXPORTPATH = os.path.join(os.getcwd(), 'Exports')
os.makedirs(EXPORTPATH, exist_ok=True)

wandb.login()

# # Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="eml-labs",
    # Set the wandb project where this run will be logged.
    project="Prediction-PAF-Onset-with-ST-CoxNet",
    # Track hyperparameters and run metadata.
    config={
        "cpc_learning_rate": CPC_LR,
        "cox_learning_rate": COX_LR,
        "cpc_epochs": CPC_EPOCHS,
        "cox_epochs": COX_EPOCHS,
        "latent_size": LATENT_SIZE,
        "context_size": CONTEXT_SIZE,
        "number_of_targets_for_prediction": NUMBER_OF_TARGETS_FOR_PREDICTION,
        "number_of_heads": NUMBER_OF_HEADS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "horizons": HORIZONS,
        "sequence_length": SEQUENCE_LENGTH,
        "validation_ratio": VALIDATION_RATIO,
        "batch_size": BATCH_SIZE,
        "prediction": PREDICTION,
        "notes": NOTES,
        "dataset": "IRIDA-AF",
    },
)

dataset_path = os.path.join('/home','intellisense01','EML-Labs','datasets','Data-Explorer','processed_data_60min_nsr_5min_af')
df = pd.read_csv(os.path.join(dataset_path,'200x20_extracted_rr_intervals.csv'))
logger.info(f"Total records in dataset: {len(df)}")

all_patients = np.sort(np.unique(df['patient_id'].values))
logger.info(f"Total unique patients: {len(all_patients)}")

np.random.shuffle(all_patients)

val_count = max(1, int(len(all_patients) * VALIDATION_RATIO))

val_patients = all_patients[:val_count]
train_patients = all_patients[val_count:]
logger.info(f"Training patients: {len(train_patients)}, Validation patients: {len(val_patients)}")

train_dataset = RRSequenceDataset(dataset_path,'200x20_extracted_rr_intervals.csv',train_patients, WINDOW_SIZE, STRIDE, HORIZONS, SEQUENCE_LENGTH)
rr_scaler, hrv_scaler = train_dataset.get_scalers()
val_dataset = RRSequenceDataset(dataset_path,'200x20_extracted_rr_intervals.csv',val_patients, WINDOW_SIZE, STRIDE, HORIZONS, SEQUENCE_LENGTH,rr_scaler=rr_scaler, hrv_scaler=hrv_scaler)
logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

    
# # Model & optimizer
model = CPCPreModel(num_targets=NUMBER_OF_TARGETS_FOR_PREDICTION, latent_dim=LATENT_SIZE, context_dim=CONTEXT_SIZE,number_of_heads=NUMBER_OF_HEADS).to(device)
optimizer = AdamW(model.parameters(), lr=CPC_LR)
logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")


for epoch in tqdm(range(1, CPC_EPOCHS + 1), desc=f"Epochs", unit="epoch"):
    model.train()
    running_loss = 0.0
    batch_count = 0
    for rr_windows, hrv_targets in train_loader:
        rr_windows = rr_windows.to(device)
        hrv_targets = hrv_targets.to(device)
        optimizer.zero_grad()
        loss,loss_1,loss_2,loss_4 = training_step(model, rr_windows, hrv_targets,epoch,CPC_EPOCHS)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1
        
    train_loss = running_loss / batch_count
    val_loss,val_loss_1,val_loss_2,val_loss_4 = validation_step(model, val_loader, device)
    run.log({
        "CPC_epoch": epoch,
        "CPC_train_loss": train_loss,
        "CPC_train_loss_1": loss_1.item(),
        "CPC_train_loss_2": loss_2.item(),
        "CPC_train_loss_4": loss_4.item(),
        "CPC_val_loss": val_loss,
        "CPC_val_loss_1": val_loss_1,
        "CPC_val_loss_2": val_loss_2,
        "CPC_val_loss_4": val_loss_4
    })
    tqdm.write(f"Epoch {epoch:02d}: Train Loss = {train_loss:.6f} Validation Loss = {val_loss:.6f} 1 : {val_loss_1:.6f} 2 : {val_loss_2:.6f} 4 : {val_loss_4:.6f}")
    logger.info(f"Epoch {epoch:02d}: Train Loss = {train_loss:.6f} Validation Loss = {val_loss:.6f} 1 : {val_loss_1:.6f} 2 : {val_loss_2:.6f} 4 : {val_loss_4:.6f}")


torch.save(model.state_dict(), os.path.join(EXPORTPATH, f'cpc_pre_model_epoch_{CPC_EPOCHS}.pth'))

artifact = wandb.Artifact(name=f'cpc_pre_model_epoch_{CPC_EPOCHS}', type='model')
artifact.add_file(local_path=os.path.join(EXPORTPATH, f'cpc_pre_model_epoch_{CPC_EPOCHS}.pth'), name=f'cpc_pre_model_epoch_{CPC_EPOCHS}.pth')
run.log_artifact(artifact)


train_cox_dataset = RRSequenceCoxDataset(train_patients, dataset_path, '200x20_extracted_rr_intervals.csv', limit_time_to_event=0.5, seq_len=SEQUENCE_LENGTH, rr_scaler=rr_scaler)
validation_cox_dataset = RRSequenceCoxDataset(val_patients, dataset_path, '200x20_extracted_rr_intervals.csv', limit_time_to_event=0.5, seq_len=SEQUENCE_LENGTH, rr_scaler=rr_scaler)
logger.info(f"Cox Train samples: {len(train_cox_dataset)}, Cox Validation samples: {len(validation_cox_dataset)}")

train_cox_loader = DataLoader(train_cox_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4)
validation_cox_loader = DataLoader(validation_cox_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4)
logger.info(f"Cox Train batches: {len(train_cox_loader)}, Cox Validation batches: {len(validation_cox_loader)}")

encoder = model.encoder
context = model.context
logger.info(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}, Context parameters: {sum(p.numel() for p in context.parameters())}")

# for param in encoder.parameters():
#     param.requires_grad = False

# for param in context.parameters():
#     param.requires_grad = False

cox_model = DeepSurvCox(encoder=encoder, context=context, context_dim=CONTEXT_SIZE, latent_dim=LATENT_SIZE).to(device)
logger.info(f"Cox model initialized with {sum(p.numel() for p in cox_model.parameters())} parameters")

cox_optimizer = AdamW(cox_model.parameters(), lr=COX_LR)
cox_loss_fn = DeepSurvLoss()
c_index = CIndex()

for epoch in tqdm(range(1, COX_EPOCHS + 1), desc=f"Cox Epochs", unit="epoch"):
    cox_model.train()
    running_loss = 0.0
    batch_count = 0
    for rr_windows, time_to_event, event in train_cox_loader:
        rr_windows = rr_windows.to(device)
        time_to_event = time_to_event.to(device)
        event = event.to(device)

        cox_optimizer.zero_grad()
        risk,_,_ = cox_model(rr_windows)  # [B]
        loss = cox_loss_fn(risk, time_to_event, event)
        loss.backward()
        cox_optimizer.step()

        running_loss += loss.item()
        batch_count += 1

    
    train_loss = running_loss / batch_count

    train_risk = []
    train_time = []
    train_event = []

    cox_model.eval()

    with torch.no_grad():
        for rr_windows, time_to_event, event in train_cox_loader:
            rr_windows = rr_windows.to(device)

            risk,_,_ = cox_model(rr_windows)

            train_risk.append(risk.cpu())
            train_time.append(time_to_event)
            train_event.append(event)

    train_risk = torch.cat(train_risk)
    train_time = torch.cat(train_time)
    train_event = torch.cat(train_event)

    times, baseline_cumhaz = compute_baseline_hazard(
        train_risk,
        train_time,
        train_event
    )

    all_risk = []
    all_time = []
    all_event = []

    pred_times = []
    actual_times = []

    encoder_embeddings = []
    context_embeddings = []

    for rr_windows, time_to_event, event in validation_cox_loader:
        rr_windows = rr_windows.to(device)
        time_to_event = time_to_event.to(device)
        event = event.to(device)
        all_time.append(time_to_event)
        all_event.append(event)
        cox_model.eval()
        actual_times.append(time_to_event.cpu().tolist())
        with torch.no_grad():
            risk,c_seq,z_seq = cox_model(rr_windows)
            risk = risk.cpu()
            encoder_embeddings.append(z_seq[:,-1,:].cpu())
            context_embeddings.append(c_seq[:,-1,:].cpu())
            local_pred_times = []
            for r in risk:
                t = predict_median_survival(r, times, baseline_cumhaz)
                local_pred_times.append(t.cpu().item())
            pred_times.append(local_pred_times)
            all_risk.append(risk.cpu())

    all_risk = torch.cat(all_risk)
    all_time = torch.cat(all_time)
    all_event = torch.cat(all_event)

    encoder_embeddings = torch.cat(encoder_embeddings)
    context_embeddings = torch.cat(context_embeddings)

    time_np = all_time.cpu().numpy()
    event_np = all_event.cpu().numpy()
    encoder_embeddings_np = encoder_embeddings.cpu().numpy()
    context_embeddings_np = context_embeddings.cpu().numpy()

    pca_emb = PCA(n_components=2, random_state=42)
    pca_cont = PCA(n_components=2, random_state=42)
    context_embedding_2d = pca_cont.fit_transform(context_embeddings_np)
    embedding_2d = pca_emb.fit_transform(encoder_embeddings_np)

    fig = plt.figure(figsize=(12,12))
    fig.suptitle(f"Cox Model Predictions at Epoch {epoch}", fontsize=16)
    gs = fig.add_gridspec(3, 2)

    # ---- Plot 1 : Scatter ----
    ax1 = fig.add_subplot(gs[0,0])
    ax1.scatter(actual_times, pred_times, alpha=0.3, s=8)

    max_t = max(np.max(actual_times), np.max(pred_times))
    ax1.plot([0, max_t], [0, max_t], 'r--', label="Perfect prediction")

    ax1.set_xlabel("Actual Time-to-Event")
    ax1.set_ylabel("Predicted Time-to-Event")
    ax1.set_title(f"Prediction vs Actual")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---- Plot 2 : Hexbin Density ----
    ax2 = fig.add_subplot(gs[0,1])
    hb = ax2.hexbin(actual_times, pred_times, gridsize=50, cmap='Blues')
    fig.colorbar(hb, ax=ax2, label="Density")

    ax2.set_xlabel("Actual Time")
    ax2.set_ylabel("Predicted Time")
    ax2.set_title("Prediction Density")
    ax2.grid(alpha=0.2)

    # ---- Plot 3 : Risk Distribution (Full Width) ----
    ax3 = fig.add_subplot(gs[1,:])

    risk_np = all_risk.numpy()

    ax3.hist(risk_np, bins=40, color="steelblue", alpha=0.75, edgecolor="black")

    # Add statistics
    mean_risk = np.mean(risk_np)
    median_risk = np.median(risk_np)

    ax3.axvline(mean_risk, color='red', linestyle='--', label=f"Mean = {mean_risk:.2f}")
    ax3.axvline(median_risk, color='green', linestyle='--', label=f"Median = {median_risk:.2f}")

    ax3.set_xlabel("Predicted Risk Score")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Risk Score Distribution")
    ax3.legend()
    ax3.grid(alpha=0.3)

    colors = np.where(event_np == 1, "red", "blue")
    # ---- Plot 4 : Encoder t-SNE ----
    ax4 = fig.add_subplot(gs[2,0])

    scatter1 = ax4.scatter(
        embedding_2d[:,0],
        embedding_2d[:,1],
        c=colors,
        alpha=0.7,
        s=8
    )

    ax4.set_title("Encoder Embedding t-SNE")
    ax4.set_xlabel("t-SNE 1")
    ax4.set_ylabel("t-SNE 2")


    # ---- Plot 5 : Context t-SNE ----
    ax5 = fig.add_subplot(gs[2,1])

    scatter2 = ax5.scatter(
        context_embedding_2d[:,0],
        context_embedding_2d[:,1],
        c=colors,
        alpha=0.7,
        s=8
    )

    ax5.set_title("Context Embedding t-SNE")
    ax5.set_xlabel("t-SNE 1")
    ax5.set_ylabel("t-SNE 2")

    # fig.colorbar(scatter2, ax=ax5, label="Event")

    plt.tight_layout(rect=[0,0,1,0.96])

    val_c_index = CIndex.calculate(all_risk, all_time, all_event)
    run.log({
        "Cox_epoch": epoch,
        "Cox_train_loss": train_loss,
        "Cox_val_c_index": val_c_index,
        "prediction_plot": wandb.Image(fig)
    })

    plt.close(fig)  
    tqdm.write(f"Cox Epoch {epoch:02d}: Train Loss = {train_loss:.6f} Validation C-Index = {val_c_index:.6f}")
    logger.info(f"Cox Epoch {epoch:02d}: Train Loss = {train_loss:.6f} Validation C-Index = {val_c_index:.6f}")

cox_model_path = os.path.join(EXPORTPATH, f'cox_model_epoch_{COX_EPOCHS}.pth')
torch.save(cox_model.state_dict(), cox_model_path)

cox_artifact = wandb.Artifact(name=f'cox_model_epoch_{COX_EPOCHS}', type='model')
cox_artifact.add_file(local_path=cox_model_path, name=f'cox_model_epoch_{COX_EPOCHS}.pth')
run.log_artifact(cox_artifact)

logger.info("Training complete.")
run.finish()