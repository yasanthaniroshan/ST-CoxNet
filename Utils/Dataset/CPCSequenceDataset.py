import torch
from torch.utils.data import Dataset
import pickle
import hashlib
import json
import os
import h5py
import numpy as np


class CPCSequenceDataset(Dataset):
    """Sliding windows of K consecutive SR segments from the same patient.

    Returns sequences for trajectory-level classification: the label is
    the time-bin of the *last* segment in each window.
    """

    def __init__(
        self,
        processed_dataset_path: str,
        afib_length: int,
        sr_length: int,
        number_of_windows_in_segment: int,
        stride: int,
        window_size: int,
        bin_edges: list,
        seq_len: int = 16,
        seq_stride: int = 1,
        validation_split: float = 0.15,
        train: bool = True,
    ):
        dataset_prop = {
            "dataset_name": "IRIDIA AFIB Dataset",
            "AFIB_length_seconds": afib_length,
            "SR_length_seconds": sr_length,
            "window_size": window_size,
            "segment_size": number_of_windows_in_segment * window_size,
            "stride": stride,
            "validation_split": validation_split,
            "scaler": "RobustScaler",
        }

        dataset_string = json.dumps(dataset_prop, sort_keys=True)
        file_name_hash = hashlib.sha256(dataset_string.encode()).hexdigest()[:32]

        dataset_props_path = os.path.join(
            processed_dataset_path, f"{file_name_hash}.pkl"
        )
        if not os.path.exists(dataset_props_path):
            raise FileNotFoundError("Dataset properties file not found.")

        dataset_prop_loaded = pickle.load(open(dataset_props_path, "rb"))
        if dataset_prop_loaded != dataset_prop:
            raise ValueError("Dataset properties mismatch.")

        split = "train" if train else "validation"
        file_path = os.path.join(
            processed_dataset_path, f"{file_name_hash}_{split}.h5"
        )

        with h5py.File(file_path, "r") as f:
            data = f["segments"][:]
            labels = f["labels"][:]
            times = f["times"][:]

        sr_mask = labels == -1
        data = data[sr_mask]
        times = times[sr_mask]

        patient_ids = self._recover_patient_ids(times)
        n_patients = patient_ids.max() + 1

        bin_edges = np.array(bin_edges, dtype=np.float64)
        bin_labels = np.clip(np.digitize(times, bin_edges) - 1, 0, len(bin_edges) - 2)
        num_bins = len(bin_edges) - 1

        self.all_data = torch.tensor(data, dtype=torch.float32)
        self.all_bin_labels = torch.tensor(bin_labels, dtype=torch.long)
        self.all_times = torch.tensor(times, dtype=torch.float32)
        self.patient_ids = patient_ids
        self.num_classes = num_bins
        self.bin_edges = bin_edges
        self.seq_len = seq_len

        self.sequences = self._build_sequences(
            patient_ids, n_patients, len(data), seq_len, seq_stride
        )

    @staticmethod
    def _recover_patient_ids(sr_times):
        n = len(sr_times)
        patient_ids = np.zeros(n, dtype=np.int32)
        pid = 0
        for i in range(1, n):
            if sr_times[i] > sr_times[i - 1] + 1e-6:
                pid += 1
            patient_ids[i] = pid
        return patient_ids

    @staticmethod
    def _build_sequences(patient_ids, n_patients, n_total, seq_len, seq_stride):
        """Build (start_idx, end_idx) pairs for sliding windows within each patient."""
        sequences = []
        for pid in range(n_patients):
            idxs = np.where(patient_ids == pid)[0]
            if len(idxs) < seq_len:
                continue
            for start in range(0, len(idxs) - seq_len + 1, seq_stride):
                seq_idxs = idxs[start : start + seq_len]
                sequences.append(seq_idxs)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_idxs = self.sequences[idx]
        rr_seq = self.all_data[seq_idxs]
        label = self.all_bin_labels[seq_idxs[-1]]
        return rr_seq, label, seq_idxs
