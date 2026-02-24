CSV_PATH = "/Users/yasantha-mac/FYP/ST-CoxNet/Utils/Dataset/AFDB/CSV_Files"
from Utils.Dataset.RRSequenceCSVDataset import RRSequenceCSVDataset

dataset = RRSequenceCSVDataset(
    rri_csv_path=f"{CSV_PATH}/rri_windows.csv",
    features_csv_path=f"{CSV_PATH}/hrv_features.csv",
    seq_len=10,
    horizons=[1, 4, 8]
)

for i in range(5):
    rri_windows, current_hrv, future_hrvs = dataset[i]
    print(f"RRI Windows Shape: {rri_windows.shape}")
    print(f"Current HRV Shape: {current_hrv.shape}")
    print(f"Future HRVs Shape: {future_hrvs.shape}")