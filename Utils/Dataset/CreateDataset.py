import os
import torch
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

def load_record_details(patient_list: list, dataset_path: str,afib_length:int,sr_length:int):
    records = []
    for patient in patient_list:
        record_dir = os.path.join(dataset_path, patient)
        ecg_csv_file = f"{patient}_ecg_labels.csv"
        ecg_df = pd.read_csv(os.path.join(record_dir, ecg_csv_file))
        for idx, row in ecg_df.iterrows():
            if (row['start_file_index'] == row['end_file_index']) and (row['af_duration'] >= afib_length) and (row['nsr_before_duration'] >= sr_length+6000):
                records.append(
                    {
                        "patient": patient,
                        "record_index": idx,
                        "start_file_index": row['start_file_index'],
                        "end_file_index": row['end_file_index']
                    }
                )
    return records

def load_rr_data(patient_data: dict,dataset_path:str,afib_length:int,sr_length:int):

    patient = patient_data['patient']
    record_index = patient_data['record_index']
    record_dir = os.path.join(dataset_path, patient)

    rr_csv_file = f"{patient}_rr_labels.csv"
    rr_df = pd.read_csv(os.path.join(record_dir, rr_csv_file))

    row = rr_df.loc[record_index]

    if row.empty:
        raise ValueError("RR index mapping not found")

    rr_start_index = row['start_rr_index']
    rr_end_index = row['end_rr_index']
    file_index = row['start_file_index']


    with h5py.File(os.path.join(record_dir, f"{patient}_rr_{file_index:02d}.h5"), 'r') as f:
        rr_data = f['rr'][:]

    # AF segment
    afib_segment = rr_data[rr_start_index:rr_end_index]
    afib_cum_sum = np.cumsum(afib_segment) / 1000
    index = np.searchsorted(afib_cum_sum, afib_length)
    afib_segment = afib_segment[:index]

    # SR segment before AF
    sr_segment = rr_data[:rr_start_index][::-1]
    sr_cum_sum = np.cumsum(sr_segment) / 1000
    index = np.searchsorted(sr_cum_sum, sr_length)
    sr_segment = sr_segment[:index][::-1]

    rr_data_combined = np.concatenate([sr_segment, afib_segment])
    afib_start_index = len(sr_segment)
    tqdm.write(f"{patient}: Total AFIB duration in seconds: {np.sum(afib_segment)/1000:.2f} length : {len(afib_segment)}, Total SR duration in seconds: {np.sum(sr_segment)/1000:.2f} length: {len(sr_segment)}")
    return rr_data_combined, afib_start_index


def find_time_to_event(segments: list, stride: int):
    time_so_far = 0
    event_times = []
    event_times.append(time_so_far)
    for segment in reversed(segments[:-1]):
        time_of_window = sum(segment.flatten()[:stride])
        time_so_far += time_of_window
        event_times.append(time_so_far)
    return event_times[::-1]

def segment_rr_data(rr_data: np.ndarray, afib_start_index: int,segment_size:int, stride:int, window_size:int):
    segments = []
    labels = []
    times = []
    is_time_calculated = False
    total_length = len(rr_data)
    for start in range(0, total_length - segment_size + 1, stride):
        end = start + segment_size
        segment = rr_data[start:end]
        segment = np.array(segment)
        segment = segment.reshape(-1, window_size)
        if start < afib_start_index and end <= afib_start_index:
            label = -1 # Pure SR segment
        elif start < afib_start_index and end > afib_start_index:
            label = 0 # Segment contains AFIB
        elif start >= afib_start_index and end > afib_start_index:
            label = 1 # Segment is entirely within AFIB
        segments.append(segment)
        labels.append(label)
        if is_time_calculated:
            times.append(0) # For segments after AFIB start, time to event is 0
        elif end >= afib_start_index and not is_time_calculated:
            time_to_event = find_time_to_event(segments, stride)
            times.extend(time_to_event)
            is_time_calculated = True
    print(f"Length of segments: {len(segments)}, Length of labels: {len(labels)}, Length of times: {len(times)} max time to event seconds: {max(times)/1000:.2f}")
    return np.array(segments), np.array(labels), np.array(times)



def create_dataset(dataset_path:str,afib_length:int,sr_length:int,number_of_windows_in_segment:int, stride:int, window_size:int,validation_split:float=0.15):
    try:
        segment_size = number_of_windows_in_segment*window_size # Number of RR intervals in each segment (20 windows)
        dataset_prop = {
            "dataset_name": "IRIDIA AFIB Dataset",
            "AFIB_length_seconds": afib_length,
            "SR_length_seconds": sr_length,
            "window_size": window_size,
            "segment_size": segment_size,
            "stride": stride,
            "validation_split": validation_split,
            "scaler":"RobustScaler",
        }

        dataset_string = json.dumps(dataset_prop, sort_keys=True)
        file_name_hash = hashlib.sha256(dataset_string.encode()).hexdigest()[:32]
        pickle.dump(dataset_prop, open(os.path.join(EXPORT_PATH, f"{file_name_hash}.pkl"), "wb"))

        print(f"Dataset properties: {dataset_prop}")
        print(f"Dataset file name hash: {file_name_hash}")
        patient_list = os.listdir(dataset_path)
        patient_list = sorted(patient_list)
        random.shuffle(patient_list)

        print(f"Total patients found: {len(patient_list)}")
        scaler = RobustScaler()

        train_patients = patient_list[:int(len(patient_list)*(1-validation_split))]
        validation_patients = patient_list[int(len(patient_list)*(1-validation_split)):]
        train_records = load_record_details(train_patients, dataset_path,afib_length,sr_length)
        validation_records = load_record_details(validation_patients, dataset_path,afib_length,sr_length)

        stats = {}
        max_time_to_event = 0
        segments = []
        labels = []
        times = []
        pbar = tqdm(total=len(train_records), desc="Processing train Patients")
        for patient in train_records:
            pbar.set_description(f"Processing train patient: {patient['patient']}")
            rr_data,afib_start_index = load_rr_data(patient, dataset_path,afib_length,sr_length)
            segment, label, time = segment_rr_data(rr_data, afib_start_index, segment_size, stride, window_size)
            if len(segment) == 0:
                pbar.write(f"Skipping patient {patient['patient']}, RR length: {len(rr_data)}")
            segments.append(segment)
            labels.append(label)
            times.append(time)
            pbar.update(1)

        segments = np.concatenate(segments, axis=0)
        labels = np.concatenate(labels, axis=0)
        times = np.concatenate(times, axis=0)
        max_time_to_event = max(times)
        times = times/max_time_to_event
        print(f"Max time to event in hours: {max(times):.2f} Min time to event in hours: {min(times):.2f}")
        segments_reshaped = segments.reshape(-1, window_size)
        segments_scaled = scaler.fit_transform(segments_reshaped)
        segments = segments_scaled.reshape(segments.shape)

        with h5py.File(os.path.join(EXPORT_PATH, f"{file_name_hash}_train.h5"), 'w') as f:
            f.create_dataset('segments', data=segments.astype('float32'), compression="gzip", chunks=True)
            f.create_dataset('labels', data=labels.astype('int8'), compression="gzip")
            f.create_dataset('times', data=times.astype('float32'), compression="gzip")

        stats['train'] = {
            "total_segments": len(labels),
            "afib_segments": int(np.sum(labels == 1)),
            "sr_segments": int(np.sum(labels == -1)),
            "mixed_segments": int(np.sum(labels == 0)),
        }

        segments = []
        labels = []
        times = []
        pbar = tqdm(total=len(validation_records), desc="Processing validation Patients")
        for patient in validation_records:
            pbar.set_description(f"Processing validation patient: {patient['patient']}")
            rr_data,afib_start_index = load_rr_data(patient, dataset_path,afib_length,sr_length)
            segment, label, time = segment_rr_data(rr_data, afib_start_index, segment_size, stride, window_size)
            if len(segment) == 0:
                pbar.write(f"Skipping patient {patient['patient']}, RR length: {len(rr_data)}")
            segments.append(segment)
            labels.append(label)
            times.append(time)
            pbar.update(1)

        segments = np.concatenate(segments, axis=0)
        labels = np.concatenate(labels, axis=0)
        times = np.concatenate(times, axis=0)
        times = times / max_time_to_event if max_time_to_event > 0 else times
        print(f"Max time to event in hours: {max(times):.2f} Min time to event in hours: {min(times):.2f}")
        segments_reshaped = segments.reshape(-1, window_size)
        segments_scaled = scaler.transform(segments_reshaped)
        segments = segments_scaled.reshape(segments.shape)

        with h5py.File(os.path.join(EXPORT_PATH, f"{file_name_hash}_validation.h5"), 'w') as f:
            f.create_dataset('segments', data=segments.astype('float32'), compression="gzip", chunks=True)
            f.create_dataset('labels', data=labels.astype('int8'), compression="gzip")
            f.create_dataset('times', data=times.astype('float32'), compression="gzip")
        stats['validation'] = {
            "total_segments": int(len(labels)),
            "afib_segments": int(np.sum(labels == 1)),
            "sr_segments": int(np.sum(labels == -1)),
            "mixed_segments": int(np.sum(labels == 0)),
        }

        with open(os.path.join(EXPORT_PATH, f"{file_name_hash}_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Dataset creation completed. Stats: {stats}")
    except Exception as e:
        print(f"Error during dataset creation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    create_dataset(
        dataset_path="/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1",
        afib_length=60*60,
        sr_length=int(1.5*60*60),
        number_of_windows_in_segment=10,
        stride=20,
        window_size=100,
    )

