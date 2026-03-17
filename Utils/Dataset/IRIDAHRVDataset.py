from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import joblib


class HRVSequenceDataset(Dataset):
    def __init__(self,dataset_path,csv_file_name,patient_list, window_size:int, stride:int, horizons:list, seq_len:int, hrv_scaler:RobustScaler=None,features_list:list=None):

        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons
        self.seq_len = seq_len
        self.samples = []
        self.hrv_scaler = hrv_scaler if hrv_scaler is not None else RobustScaler()
        self.features_list = features_list if features_list is not None else ['RMSSD', 'pNN50', 'SDNN', 'alpha_1', 'sample_entropy', 'approximate_entropy']

        # Each sample contains seq_len historical RR windows and one HRV target per horizon.
        # Rows in the CSV already correspond to pre-windowed segments, so we index by rows.
        last_target_offset = (seq_len - 1) + max(horizons)

        df = pd.read_csv(os.path.join(dataset_path, csv_file_name))
        df = df[df['patient_id'].isin(patient_list)]
        df['file_name'] = df['patient_id'].astype(str) + '_' + df['episode_id'].astype(str) + '_' + df['segment_id'].astype(str) + '.npy'
        all_files = df['file_name'].tolist()
        num_of_segments = [len(df[df['patient_id'] == patient]) for patient in patient_list]
        total_segments = sum(max(0, n - last_target_offset) for n in num_of_segments)
        pbar = tqdm(total=total_segments, desc="Processing patients", unit="segment")
        for patient in patient_list:
            patient_df = df[df['patient_id'] == patient]
            total_segments = len(patient_df)
            for i in range(0, total_segments - last_target_offset, 1):
                hrv_windows = []
                hrvs = []
                for j in range(i, i + seq_len):
                    hrv_window = patient_df.iloc[j][self.features_list].values.astype(np.float32)
                    hrv_windows.append(hrv_window)

                for h in horizons:
                    target_idx = i + seq_len - 1 + h
                    hrv_features = patient_df.iloc[target_idx][self.features_list].values.astype(np.float32)
                    hrvs.append(hrv_features)
                self.samples.append((hrv_windows, hrvs))
                pbar.update(1)
        pbar.close()
        # Fit scalers on the entire dataset if not provided
        df_for_dataset = df[df['patient_id'].isin(patient_list)]
        if hrv_scaler is None:
            data = df_for_dataset[self.features_list].values.astype(np.float32)
            self.hrv_scaler.fit(data)

        hrv_windows_np = np.array([sample[0] for sample in self.samples]).reshape(-1, len(self.features_list))
        hrvs_np = np.array([sample[1] for sample in self.samples]).reshape(-1, len(self.features_list))
        np.memmap(os.path.join(dataset_path, 'hrv_windows.npy'), dtype='float32', mode='w+', shape=hrv_windows_np.shape)[:] = hrv_windows_np
        np.memmap(os.path.join(dataset_path, 'hrvs.npy'), dtype='float32', mode='w+', shape=hrvs_np.shape)[:] = hrvs_np


    def get_scalers(self):
        return self.hrv_scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hrv_windows, hrvs = self.samples[idx]
        hrv_windows_transformed = self.hrv_scaler.transform(hrv_windows)
        hrvs_transformed = self.hrv_scaler.transform(hrvs)
        return torch.tensor(hrv_windows_transformed, dtype=torch.float32), torch.tensor(hrvs_transformed, dtype=torch.float32)