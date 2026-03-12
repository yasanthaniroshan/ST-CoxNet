from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

class RRSequenceDataset(Dataset):
    def __init__(self,dataset_path,csv_file_name,patient_list, window_size:int, stride:int, horizons:list, seq_len:int,rr_scaler:RobustScaler=None, hrv_scaler:RobustScaler=None):

        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons
        self.seq_len = seq_len
        self.samples = []
        self.rr_scaler = rr_scaler if rr_scaler is not None else RobustScaler()
        self.hrv_scaler = hrv_scaler if hrv_scaler is not None else RobustScaler()

        # Each sample contains seq_len historical RR windows and one HRV target per horizon.
        # Rows in the CSV already correspond to pre-windowed segments, so we index by rows.
        last_target_offset = (seq_len - 1) + max(horizons)

        df = pd.read_csv(os.path.join(dataset_path, csv_file_name))
        df = df[df['patient_id'].isin(patient_list)]
        df['file_name'] = df['patient_id'].astype(str) + '_' + df['episode_id'].astype(str) + '_' + df['segment_id'].astype(str) + '.npy'
        all_files = df['file_name'].tolist()
        rr_data = dict()
        for file_name in tqdm(all_files, desc="Preloading RR windows", unit="file"):
            rr_data[file_name] = np.load(os.path.join(dataset_path, file_name))
        num_of_segments = [len(df[df['patient_id'] == patient]) for patient in patient_list]
        total_segments = sum(max(0, n - last_target_offset) for n in num_of_segments)
        pbar = tqdm(total=total_segments, desc="Processing patients", unit="segment")
        for patient in patient_list:
            patient_df = df[df['patient_id'] == patient]
            total_segments = len(patient_df)
            for i in range(0, total_segments - last_target_offset, 1):
                rr_windows = []
                hrvs = []
                for j in range(i, i + seq_len):
                    rr_window = rr_data[patient_df.iloc[j]['file_name']]
                    rr_windows.append(rr_window)

                for h in horizons:
                    target_idx = i + seq_len - 1 + h
                    hrv_features = patient_df.iloc[target_idx][['RMSSD', 'pNN50', 'SDNN', 'alpha_1', 'sample_entropy', 'approximate_entropy']].values.astype(np.float32)
                    hrvs.append(hrv_features)
                self.samples.append((rr_windows, hrvs))
                pbar.update(1)
        pbar.close()
        # Fit scalers on the entire dataset if not provided
        if rr_scaler is None or hrv_scaler is None:
            all_rr_windows = np.concatenate([np.array(sample[0]) for sample in self.samples], axis=0)  # [total_samples, window_size]
            all_hrvs = np.concatenate([np.array(sample[1]) for sample in self.samples], axis=0)  # [total_samples, num_metrics]
            self.rr_scaler.fit(all_rr_windows)
            self.hrv_scaler.fit(all_hrvs)

    def get_scalers(self):
        return self.rr_scaler, self.hrv_scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rr_windows, hrvs = self.samples[idx]
        rr_windows_transformed = self.rr_scaler.transform(rr_windows)
        hrvs_transformed = self.hrv_scaler.transform(hrvs)
        return torch.tensor(rr_windows_transformed, dtype=torch.float32), torch.tensor(hrvs_transformed, dtype=torch.float32)