import torch
from torch.utils.data import Dataset, Sampler
import pickle
import hashlib
import json
import os
import h5py
import random
from collections import defaultdict


class CPCTemporalDataset(Dataset):
    """
    Dataset for CPC + within-patient temporal ranking.

    Returns (segment, label, tte_seconds, patient_id) per item.
    Use with PatientBatchSampler for patient-grouped batches.
    """

    def __init__(
        self,
        processed_dataset_path,
        afib_length,
        sr_length,
        number_of_windows_in_segment,
        stride,
        window_size,
        validation_split=0.15,
        train=True,
        sr_only=False,
    ):
        dataset_prop = {
            "dataset_name": "IRIDIA AFIB Temporal Dataset v2",
            "AFIB_length_seconds": afib_length,
            "SR_length_seconds": sr_length,
            "window_size": window_size,
            "segment_size": number_of_windows_in_segment * window_size,
            "stride": stride,
            "validation_split": validation_split,
            "scaler": "RobustScaler",
        }

        dataset_string = json.dumps(dataset_prop, sort_keys=True)
        file_hash = hashlib.sha256(dataset_string.encode()).hexdigest()[:32]

        pkl_path = os.path.join(processed_dataset_path, f"{file_hash}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Dataset not found. Run CreateTemporalDataset.py first.\n"
                f"Expected: {pkl_path}"
            )

        props = pickle.load(open(pkl_path, "rb"))
        if props != dataset_prop:
            raise ValueError("Dataset properties mismatch.")

        suffix = "train" if train else "validation"
        h5_path = os.path.join(processed_dataset_path, f"{file_hash}_{suffix}.h5")

        with h5py.File(h5_path, "r") as f:
            data = f["segments"][:]
            labels = f["labels"][:]
            times = f["times"][:]
            patient_ids = f["patient_ids"][:]
            self.max_tte = float(f.attrs.get("max_tte_seconds", 1.0))

        if sr_only:
            mask = labels == -1
            data = data[mask]
            labels = labels[mask]
            times = times[mask]
            patient_ids = patient_ids[mask]

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.patient_ids = torch.tensor(patient_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.times[idx], self.patient_ids[idx]


class PatientBatchSampler(Sampler):
    """
    Yields batches of P patients × K segments per patient.

    Batch structure is guaranteed: indices [0:K] belong to patient 0,
    [K:2K] to patient 1, etc. This enables efficient within-patient
    temporal ranking via reshape(P, K, ...).

    Each epoch samples `batches_per_epoch` random batches. Patients and
    segments are re-sampled every batch for diversity.
    """

    def __init__(self, patient_ids, P, K, batches_per_epoch=50):
        self.P = P
        self.K = K
        self.batches_per_epoch = batches_per_epoch

        self.patient_to_indices = defaultdict(list)
        for idx, pid in enumerate(patient_ids.tolist()):
            self.patient_to_indices[pid].append(idx)

        self.patients = list(self.patient_to_indices.keys())
        if len(self.patients) < P:
            raise ValueError(
                f"Need at least P={P} patients, got {len(self.patients)}"
            )

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = random.sample(self.patients, self.P)
            batch = []
            for pid in chosen:
                pool = self.patient_to_indices[pid]
                if len(pool) >= self.K:
                    batch.extend(random.sample(pool, self.K))
                else:
                    batch.extend(random.choices(pool, k=self.K))
            yield batch

    def __len__(self):
        return self.batches_per_epoch
