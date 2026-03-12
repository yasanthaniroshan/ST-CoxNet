path = "Utils/Dataset/AFDB/CSV_Files"


import os
import pandas as pd

def check_RMSSD_values(path):
    df = pd.read_csv(os.path.join(path, "hrv_features.csv"))

    if "HRV_RMSSD" in df.columns:
        nan_rows = df[df["HRV_RMSSD"].isna()]
        if not nan_rows.empty:
            for _, row in nan_rows.iterrows():
                segment_name = row.get("Segment_Name", "N/A")
                start_idx = row.get("Start_Idx", "N/A")
                end_idx = row.get("End_Idx", "N/A")

                print(f"Segment: {segment_name}, Start: {start_idx}, End: {end_idx}")
        else:
            print("No NaN values found in the 'HRV_RMSSD' column.")

    else:
            print("Column 'HRV_RMSSD' not found in the CSV file.")
if __name__ == "__main__":
    check_RMSSD_values(path)