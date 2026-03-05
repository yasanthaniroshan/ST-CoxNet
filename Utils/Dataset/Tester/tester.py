
import sys
import os
segment_folder = "Utils/Dataset/AFDB/Extracted_Segments/"
segment_file = "04043_ep02"  # Changed to existing file
segment_path = os.path.join(segment_folder, segment_file)

import numpy as np
import wfdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
import neurokit2 as nk

# Read the WFDB record
record = wfdb.rdrecord(segment_path)
ecg_signal = record.p_signal[:, 0]  # Get the first channel (usually ECG)
sampling_rate = record.fs

print(f"Loaded ECG signal with {len(ecg_signal)} samples at {sampling_rate} Hz")

peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
r_peaks = peaks["ECG_R_Peaks"].values
peak_indices = np.where(r_peaks == 1)[0]
rri = np.diff(peak_indices) * 1000 / sampling_rate  # Use actual sampling rate
print(f"R-peak indices: {peak_indices}")
print(f"RR intervals (ms): {rri[:50]}")

rmssd = np.sqrt(np.mean(np.diff(rri[:50]) ** 2))
print(f"RMSSD of first 50 RR intervals: {rmssd}")






