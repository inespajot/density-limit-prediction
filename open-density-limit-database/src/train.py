import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tensorflow import keras

from src.preprocessing import features, split_by_discharge, scale_features, make_windows
from src.models import build_cnn

sys.stdout = open("../outputs/models.txt", "w")
DATA_PATH = Path("../DL_DataFrame.csv")
OUTPUT_DIR = Path("../outputs")

window_size = 50
forecast_horizon = 20
test_size = 0.20
val_size = 0.20
random_state = 42
window_stride = 1


def main():

    keras.utils.set_random_seed(random_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    train_df, val_df, test_df = split_by_discharge(df, random_state=random_state)
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df)

    X_train, y_train, _, _ = make_windows(
        train_df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        window_stride=window_stride,
    )

    X_val, y_val, _, _ = make_windows(
        val_df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        window_stride=window_stride,
    )

    X_test, y_test, test_discharge_ids, test_end_times = make_windows(
        test_df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        window_stride=window_stride,
    )

    print("Training windows:", X_train.shape)
    print("Validation windows:", X_val.shape)
    print("Test windows:", X_test.shape)

    print("Training positives:", y_train.sum())
    print("Validation positives:", y_val.sum())
    print("Test positives: ", y_test.sum())

    cnn_model = build_cnn(window_size=window_size, feature_number=len(features))
