import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from Utils.FeatureExtractor.HRVMetrics.HRVFeatures import HRVFeatures
import neurokit2 as nk
import numpy as np

# Fix NumPy 2.0+ compatibility issue with NeuroKit2
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from Utils.FeatureExtractor.HRVMetrics.HRVTimeFeatures import HRVTimeFeatures

data = nk.data("bio_resting_5min_100hz")
data.head()  # Print first 5 rows
peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
hrv_time = nk.hrv_time(peaks, sampling_rate=100, show=False)
print(hrv_time.columns.tolist())
print()
print(hrv_time.head())

print(peaks)

print("\nTesting HRVTimeFeatures class when the whole ECG is given")
hrv_time_features = HRVFeatures(data["ECG"], fs=100, rri_given=False)
features = hrv_time_features.compute_nonlinear_features()
print("\nExtracted HRV Nonlinear Features:")
for key, value in features.items():
    print(f"{key}: {value}")

print("\nTesting HRVTimeFeatures class when QRS peaks are given")
r_peaks = peaks["ECG_R_Peaks"].values
peak_indices = np.where(r_peaks == 1)[0]
print(f"R-peak indices: {peak_indices} ")
rri = np.diff(peak_indices) * 1000 / 100 # Convert to miliseconds (assuming fs=100Hz)
print(f"RR intervals (ms): {rri}")
hrv_time_features = HRVFeatures(rri, fs=100, rri_given=True)
features = hrv_time_features.compute_all()
print("\nExtracted HRV Features:")
for key, value in features.items():
    print(f"{key}: {value}")

# print("\nHelper information for HRV Time-Domain Features:")
# helper_info = hrv_time_features.__helper__()
# for key, value in helper_info.items():
#     print(f"{key}: {value}")

# hrv_frequency = nk.hrv_frequency(peaks, sampling_rate=100, show=True, normalize=True)
# print("hrv_frequency columns:", hrv_frequency.columns.tolist())
# for key in hrv_frequency.columns:
#     print(f"{key}: {hrv_frequency[key].values[0]}") 

# hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=100, show=True)
# print("hrv_nonlinear columns:", hrv_nonlinear.columns.tolist())
# for key in hrv_nonlinear.columns:
#     print(f"{key}: {hrv_nonlinear[key].values[0]}")

# hrv_all = nk.hrv(peaks, sampling_rate=100, show=True)
# print("hrv_all columns:", hrv_all.columns.tolist())
# for key in hrv_all.columns:
#     print(f"{key}: {hrv_all[key].values[0]}")


# rqa = nk.hrv_rqa(peaks, sampling_rate=100, show=True)
# print("hrv_rqa columns:", rqa.columns.tolist())
# for key in rqa.columns:
#     print(f"{key}: {rqa[key].values[0]}")

# from neurokit2.hrv.hrv_utils import _hrv_format_input
# from neurokit2.signal import signal_detrend

# rri, _, _ = _hrv_format_input(peaks, sampling_rate=100)
# rri = signal_detrend(rri, method="polynomial", order=1)

# import scipy.spatial
# import numpy as np

# dists = scipy.spatial.distance.pdist(np.array([rri, rri]).T, "euclidean")
# eps = 0.5 * np.mean(dists)
# print(f"Calculated epsilon for RQA: {eps}")


# from neurokit2.complexity import complexity_rqa

# nk_rqa, nk_info = complexity_rqa(
#     rri,
#     dimension=3,
#     delay=3,
#     tolerance=60
# )

# print("RQA features from NeuroKit2:")
# for key, value in nk_rqa.items():
#     print(f"{key}: {value}")