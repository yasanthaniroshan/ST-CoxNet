from torch.utils.data import Dataset
import numpy as np
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

class HRVSequenceCoxDataset(Dataset):
    def __init__(
        self,
        patient_ids,          # list of main record names (e.g., ['p16', 'p18', ...])
        dataset_path,         # path to the .hea/.dat files
        csv_file_name,       # name of the CSV file containing patient metadata (e.g., 'patient_metadata.csv')
        limit_time_to_event=0.5, # Max time to event in hours (e.g., 0.5 hours = 30 minutes)
        seq_len=10,
        hrv_scaler=None,       # Optional scaler for HRV features (e.g., RobustScaler)
        features_list=None     # List of HRV features to use (e.g., ['RMSSD', 'pNN50', 'SDNN', 'alpha_1', 'sample_entropy', 'approximate_entropy'])
    ):
        if hrv_scaler is None:
            self.hrv_scaler = RobustScaler()
        else:
            self.hrv_scaler = hrv_scaler
        csv_path = os.path.join(dataset_path, csv_file_name)
        df = pd.read_csv(csv_path)
        self.samples = []
        sample_size = seq_len
        num_of_segments = [len(df[df['patient_id'] == patient]) for patient in patient_ids]
        total_segments = sum(num_of_segments) - len(patient_ids) * (sample_size - 1)
        pbar = tqdm(total=total_segments, desc="Processing patients", unit="segment")
        for patient_id in patient_ids:
            patient_df = df[df['patient_id'] == patient_id]
            total_segments = len(patient_df)
            for i in range(0, total_segments - sample_size + 1, 1):
                pbar.update(1)
                if patient_df.iloc[i]['EventType'] == 'AFib':
                    continue  # Skip sequences that start with an AF event
                time_to_event = patient_df.iloc[i + sample_size - 1]['TimeToEvent']/3600 # Convert seconds to hours
                if round(time_to_event,2) == 0.00:
                    continue  # Skip if time to event exceeds the limit
                event = patient_df.iloc[i + sample_size - 1]['Event']
                if time_to_event > limit_time_to_event:
                    event = 0  # Censoring: event did not occur within the limit
                    time_to_event = limit_time_to_event  # Cap time to event at the limit for censored samples
                hrv_window_indexes = range(i, i + sample_size)
                hrv_windows = []
                for idx in hrv_window_indexes:
                    hrv_window = patient_df.iloc[idx][features_list].values.astype(np.float32)
                    hrv_window_scaled = self.hrv_scaler.transform(hrv_window.reshape(1, -1)).flatten()
                    hrv_windows.append(hrv_window_scaled)
                hrv_windows = np.stack(hrv_windows)  # [seq_len, window_size]
                self.samples.append((hrv_windows, time_to_event, event))
        pbar.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hrv_windows, time_to_event, event = self.samples[idx]
        return (
            torch.tensor(hrv_windows, dtype=torch.float32),  # (T, W)
            torch.tensor(time_to_event, dtype=torch.float32),  # hours
            torch.tensor(event, dtype=torch.float32)        # 0/1
        )
