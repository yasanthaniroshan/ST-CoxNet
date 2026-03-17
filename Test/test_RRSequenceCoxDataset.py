from Utils.Dataset.CoxDataset.RRSequenceCoxDataset import RRSequenceCoxDataset
from Utils.Dataset.IRIDADataset import RRSequenceDataset

import os
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Hyperparameters
LR = 5e-4
COX_LR = 1e-3
EPOCHS = 100
LATENT_SIZE = 32
CONTEXT_SIZE = 64
NUMBER_OF_TARGETS_FOR_PREDICTION = 6
NUMBER_OF_HEADS = 3
WINDOW_SIZE = 200
STRIDE = 20
HORIZONS = [2,4,8]  # Predicting HRV differences at 2, 4, and 8 windows into the future
SEQUENCE_LENGTH = 10
VALIDATION_RATIO = 0.3
BATCH_SIZE = 1024
PREDICTION = "hrv difference"

dataset_path = os.path.join('/home','intellisense01','EML-Labs','datasets','Data-Explorer','processed_data_60min_nsr_5min_af')
df = pd.read_csv(os.path.join(dataset_path,'200x20_extracted_rr_intervals.csv'))

train_patients = sorted(df['patient_id'].unique())[:2]
print(f"Total patients: {len(train_patients)}")
print(train_patients)
train_dataset = RRSequenceDataset(dataset_path,'200x20_extracted_rr_intervals.csv',train_patients, WINDOW_SIZE, STRIDE, HORIZONS, SEQUENCE_LENGTH)
rr_scaler, hrv_scaler = train_dataset.get_scalers()
train_cox_dataset = RRSequenceCoxDataset(train_patients, dataset_path, '200x20_extracted_rr_intervals.csv', limit_time_to_event=0.5, seq_len=SEQUENCE_LENGTH, rr_scaler=rr_scaler)

print(f"Total sequence samples: {len(train_dataset)}")
print(f"Total Cox samples: {len(train_cox_dataset)}")