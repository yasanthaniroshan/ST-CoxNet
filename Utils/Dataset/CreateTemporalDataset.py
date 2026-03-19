"""
Dataset generator for within-patient temporal ranking.

Key differences from CreateDataset.py:
  - Saves patient_ids per segment (enables within-patient ranking)
  - TTE stored in raw seconds (not normalized)
  - Configurable stride (default 100 → 90% overlap instead of 98%)
  - Cleaner TTE computation: sum of RR intervals from segment end to AF onset
"""

import os
import numpy as np
import pandas as pd
import random
import h5py
from tqdm import tqdm
import pickle
import hashlib
import json
import traceback
from sklearn.preprocessing import RobustScaler

np.random.seed(42)
random.seed(42)

EXPORT_PATH = os.path.join(os.getcwd(), "processed_datasets")
os.makedirs(EXPORT_PATH, exist_ok=True)


def load_record_details(patient_list, dataset_path, afib_length, sr_length):
    records = []
    for patient in patient_list:
        record_dir = os.path.join(dataset_path, patient)
        ecg_csv_file = f"{patient}_ecg_labels.csv"
        ecg_df = pd.read_csv(os.path.join(record_dir, ecg_csv_file))
        for idx, row in ecg_df.iterrows():
            if (
                row["start_file_index"] == row["end_file_index"]
                and row["af_duration"] >= afib_length
                and row["nsr_before_duration"] >= sr_length + 6000
            ):
                records.append(
                    {
                        "patient": patient,
                        "record_index": idx,
                        "start_file_index": row["start_file_index"],
                    }
                )
    return records


def load_rr_data(patient_data, dataset_path, afib_length, sr_length):
    patient = patient_data["patient"]
    record_index = patient_data["record_index"]
    record_dir = os.path.join(dataset_path, patient)

    rr_df = pd.read_csv(os.path.join(record_dir, f"{patient}_rr_labels.csv"))
    row = rr_df.loc[record_index]
    if row.empty:
        raise ValueError("RR index mapping not found")

    rr_start_index = row["start_rr_index"]
    rr_end_index = row["end_rr_index"]
    file_index = row["start_file_index"]

    with h5py.File(
        os.path.join(record_dir, f"{patient}_rr_{file_index:02d}.h5"), "r"
    ) as f:
        rr_data = f["rr"][:]

    # AF segment (trim to afib_length seconds)
    afib_segment = rr_data[rr_start_index:rr_end_index]
    afib_cum = np.cumsum(afib_segment) / 1000
    afib_segment = afib_segment[: np.searchsorted(afib_cum, afib_length)]

    # SR segment before AF (trim to sr_length seconds)
    sr_segment = rr_data[:rr_start_index][::-1]
    sr_cum = np.cumsum(sr_segment) / 1000
    sr_segment = sr_segment[: np.searchsorted(sr_cum, sr_length)][::-1]

    rr_combined = np.concatenate([sr_segment, afib_segment])
    afib_start_index = len(sr_segment)

    tqdm.write(
        f"  {patient}: SR {len(sr_segment)} beats "
        f"({np.sum(sr_segment)/1000:.0f}s), "
        f"AF {len(afib_segment)} beats "
        f"({np.sum(afib_segment)/1000:.0f}s)"
    )
    return rr_combined, afib_start_index


def segment_and_label(rr_data, afib_start_index, segment_size, stride, window_size):
    """Segment RR data, assign labels, and compute TTE in seconds."""
    segments, labels, times = [], [], []
    total_length = len(rr_data)

    for start in range(0, total_length - segment_size + 1, stride):
        end = start + segment_size
        segment = rr_data[start:end].reshape(-1, window_size)

        if end <= afib_start_index:
            label = -1  # Pure SR
            tte_seconds = np.sum(rr_data[end:afib_start_index]) / 1000.0
        elif start < afib_start_index:
            label = 0  # Mixed (transition)
            tte_seconds = 0.0
        else:
            label = 1  # AF
            tte_seconds = 0.0

        segments.append(segment)
        labels.append(label)
        times.append(tte_seconds)

    return np.array(segments), np.array(labels), np.array(times)


def create_temporal_dataset(
    dataset_path,
    afib_length=3600,
    sr_length=5400,
    number_of_windows_in_segment=10,
    stride=100,
    window_size=100,
    validation_split=0.15,
):
    try:
        segment_size = number_of_windows_in_segment * window_size

        dataset_prop = {
            "dataset_name": "IRIDIA AFIB Temporal Dataset v2",
            "AFIB_length_seconds": afib_length,
            "SR_length_seconds": sr_length,
            "window_size": window_size,
            "segment_size": segment_size,
            "stride": stride,
            "validation_split": validation_split,
            "scaler": "RobustScaler",
        }

        dataset_string = json.dumps(dataset_prop, sort_keys=True)
        file_hash = hashlib.sha256(dataset_string.encode()).hexdigest()[:32]

        print(f"Config: stride={stride}, segment={segment_size}, split={validation_split}")
        print(f"Hash:   {file_hash}")

        patient_list = sorted(os.listdir(dataset_path))
        random.shuffle(patient_list)

        split_idx = int(len(patient_list) * (1 - validation_split))
        train_patients = patient_list[:split_idx]
        val_patients = patient_list[split_idx:]
        print(f"Patients: {len(train_patients)} train, {len(val_patients)} val")

        scaler = RobustScaler()
        patient_id_map = {}

        # ── Process split ──
        def process_split(patient_list_split, desc):
            records = load_record_details(patient_list_split, dataset_path, afib_length, sr_length)
            all_segs, all_labs, all_times, all_pids = [], [], [], []

            for rec in tqdm(records, desc=desc):
                pid_str = rec["patient"]
                if pid_str not in patient_id_map:
                    patient_id_map[pid_str] = len(patient_id_map)
                pid = patient_id_map[pid_str]

                rr_data, afib_idx = load_rr_data(rec, dataset_path, afib_length, sr_length)
                segs, labs, tts = segment_and_label(
                    rr_data, afib_idx, segment_size, stride, window_size
                )
                if len(segs) == 0:
                    tqdm.write(f"  Skipping {pid_str} (no segments)")
                    continue

                all_segs.append(segs)
                all_labs.append(labs)
                all_times.append(tts)
                all_pids.append(np.full(len(segs), pid, dtype=np.int32))

            segments = np.concatenate(all_segs)
            labels = np.concatenate(all_labs)
            times = np.concatenate(all_times)
            pids = np.concatenate(all_pids)
            return segments, labels, times, pids

        # ── Train ──
        train_segs, train_labs, train_times, train_pids = process_split(
            train_patients, "Train"
        )
        flat = train_segs.reshape(-1, window_size)
        train_segs = scaler.fit_transform(flat).reshape(train_segs.shape)
        max_tte = float(train_times.max())

        with h5py.File(os.path.join(EXPORT_PATH, f"{file_hash}_train.h5"), "w") as f:
            f.create_dataset("segments", data=train_segs.astype("float32"), compression="gzip", chunks=True)
            f.create_dataset("labels", data=train_labs.astype("int8"), compression="gzip")
            f.create_dataset("times", data=train_times.astype("float32"), compression="gzip")
            f.create_dataset("patient_ids", data=train_pids.astype("int32"), compression="gzip")
            f.attrs["max_tte_seconds"] = max_tte

        # ── Validation ──
        val_segs, val_labs, val_times, val_pids = process_split(
            val_patients, "Validation"
        )
        flat = val_segs.reshape(-1, window_size)
        val_segs = scaler.transform(flat).reshape(val_segs.shape)

        with h5py.File(os.path.join(EXPORT_PATH, f"{file_hash}_validation.h5"), "w") as f:
            f.create_dataset("segments", data=val_segs.astype("float32"), compression="gzip", chunks=True)
            f.create_dataset("labels", data=val_labs.astype("int8"), compression="gzip")
            f.create_dataset("times", data=val_times.astype("float32"), compression="gzip")
            f.create_dataset("patient_ids", data=val_pids.astype("int32"), compression="gzip")
            f.attrs["max_tte_seconds"] = max_tte

        # ── Metadata ──
        pickle.dump(dataset_prop, open(os.path.join(EXPORT_PATH, f"{file_hash}.pkl"), "wb"))

        with open(os.path.join(EXPORT_PATH, f"{file_hash}_patient_map.json"), "w") as f:
            json.dump(patient_id_map, f, indent=2)

        stats = {
            "train": {
                "total": int(len(train_labs)),
                "sr": int(np.sum(train_labs == -1)),
                "mixed": int(np.sum(train_labs == 0)),
                "afib": int(np.sum(train_labs == 1)),
                "patients": int(len(np.unique(train_pids))),
            },
            "validation": {
                "total": int(len(val_labs)),
                "sr": int(np.sum(val_labs == -1)),
                "mixed": int(np.sum(val_labs == 0)),
                "afib": int(np.sum(val_labs == 1)),
                "patients": int(len(np.unique(val_pids))),
            },
            "max_tte_seconds": max_tte,
            "stride": stride,
            "overlap_pct": round((1 - stride / segment_size) * 100, 1),
        }

        with open(os.path.join(EXPORT_PATH, f"{file_hash}_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Dataset created successfully.")
        print(f"  Train:  {stats['train']['total']:>6} segments, {stats['train']['patients']} patients")
        print(f"  Val:    {stats['validation']['total']:>6} segments, {stats['validation']['patients']} patients")
        print(f"  Stride: {stride} ({stats['overlap_pct']}% overlap)")
        print(f"  Max TTE: {max_tte:.1f}s ({max_tte/60:.1f}min)")
        print(f"  Hash:  {file_hash}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    create_temporal_dataset(
        dataset_path="/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1",
        afib_length=60 * 60,
        sr_length=int(1.5 * 60 * 60),
        number_of_windows_in_segment=10,
        stride=100,
        window_size=100,
    )
