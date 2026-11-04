"""Data preprocessing and the train / calibration / test splits."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import DATA_FILE, TEMPERATURE_COLUMNS, TEMPERATURE_DATE


def process_temp_data():
    """Prepare raw city-temperature data for spherical harmonic fitting."""

    # Keep only one month before heavy processing
    chunks = []
    for chunk in pd.read_csv(
        DATA_FILE,
        delimiter=",",
        usecols=TEMPERATURE_COLUMNS,
        chunksize=50000,
    ):
        filtered = chunk[chunk["dt"] == TEMPERATURE_DATE]
        if not filtered.empty:
            chunks.append(filtered)
    data = pd.concat(chunks, ignore_index=True)

    data.rename(columns={"Latitude": "Theta", "Longitude": "Phi"}, inplace=True)
    data = data[["City", "Country", "AverageTemperature", "Theta", "Phi"]].dropna()

    def convert_latitude(lat):
        if lat[-1] == "N":
            return (90 - float(lat[:-1])) * np.pi / 180
        elif lat[-1] == "S":
            return (90 + float(lat[:-1])) * np.pi / 180
        else:
            raise ValueError("Invalid latitude format")
    
    data["Theta"] = np.array([convert_latitude(lat) for lat in data["Theta"]])
    
    def convert_longitude(long):
        if long[-1] == "W":
            return (360 - float(long[:-1])) * np.pi / 180
        elif long[-1] == "E":
            return (float(long[:-1])) * np.pi / 180
        else:
            raise ValueError("Invalid longitude format")
    
    data["Phi"] = np.array([convert_longitude(long) for long in data["Phi"]])

    print(f"Theta shape: {data['Theta'].shape}")
    print(f"Phi shape {data['Phi'].shape}")
    print(f"Temperatures: {data['AverageTemperature'].shape}")

    return data


def split_conformal_data(data, train_frac=0.6, validation_frac=0.2, cal_frac=0.1, random_seed=42):
    # Split data into train / validation / calibration / test sets for conformal prediction
    np.random.seed(random_seed)

    n = len(data)
    indices = np.random.permutation(n)

    n_train = int(train_frac * n)
    n_validation = int(validation_frac * n)
    n_cal = int(cal_frac * n)

    train_idx = indices[:n_train]
    validation_idx = indices[n_train:n_train + n_validation]
    cal_idx = indices[n_train + n_validation:n_train + n_validation + n_cal]
    test_idx = indices[n_train + n_validation + n_cal:]

    train_data = data.iloc[train_idx].reset_index(drop=True)
    validation_data = data.iloc[validation_idx].reset_index(drop=True)
    cal_data = data.iloc[cal_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(validation_data)}")
    print(f"Calibration size: {len(cal_data)}")
    print(f"Test size: {len(test_data)}")

    return train_data, validation_data, cal_data, test_data


def split_data(data, num_folds=5, random_seed=42):
    """Create true K-fold train/validation splits for cross-validation."""

    train_data_list = []
    validation_data_list = []

    kfold = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_seed
    )

    for fold, (train_idx, validation_idx) in enumerate(
        kfold.split(data), start=1
    ):

        print(
            f"Fold {fold}: "
            f"Train size = {len(train_idx)}, "
            f"Validation size = {len(validation_idx)}, "
            f"Overlap = "
            f"{len(set(train_idx).intersection(set(validation_idx)))}"
        )

        train_data = data.iloc[train_idx].reset_index(drop=True)
        validation_data = data.iloc[validation_idx].reset_index(drop=True)

        train_data_list.append(train_data)
        validation_data_list.append(validation_data)

    return train_data_list, validation_data_list
