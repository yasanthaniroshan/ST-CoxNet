from Utils.Dataset.CPCDataset import CPCDataset
import os 

if __name__ == "__main__":
    processed_dataset_path = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    afib_length = 60*60
    sr_length = 60*60
    number_of_windows_in_segment = 20
    stride = 50
    window_size = 100
    validation_split = 0.15

    dataset = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True
    )

    segment,label = dataset[0]
    print(f"Segment shape: {segment.shape}, Label: {label}")

    print(f"Loaded {len(dataset)} training segments.")