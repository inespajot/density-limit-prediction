import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = [
    "density",
    "plasma_current",
    "elongation",
    "minor_radius",
    "toroidal_B_field",
    "triangularity",]

window_size = 50
forecast_horizon = 20
test_size = 0.20
val_size = 0.20
random_state = 42

def split_by_discharge(df, test_size,val_size,random_state):
    """Split dataframe by discharge without temporal leakage"""
    discharge_targets = (df.groupby("discharge_ID")["density_limit_phase"].max().reset_index())
    train_val_ids, test_ids = train_test_split(discharge_targets["discharge_ID"],
                                               test_size = test_size,
                                               random_state = random_state,
                                               stratify = discharge_targets["density_limit_phase"])
    target_by_id = discharge_targets.set_index("discharge_ID")
    train_val_targets = target_by_id.loc[train_val_ids, "density_limit_phase"]
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_targets)
    
    train_df = df[df["discharge_ID"].isin(train_ids)].copy()
    val_df = df[df["discharge_ID"].isin(val_ids)].copy()
    test_df = df[df["discharge_ID"].isin(test_ids)].copy()

    return train_df, val_df, test_df

def scale_features(train_df,val_df, test_df):
    """Fit scaling on training data and apply it to every subset"""
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df.loc[:, features] = scaler.fit_transform(train_df[features])
    val_df.loc[:, features] = scaler.transform(val_df[features])
    test_df.loc[:, features] = scaler.transform(test_df[features])

    return train_df, val_df, test_df, scaler


def make_windows(df, window_size,forecast_horizon):
    """Convert each plasma discharge time series into sliding windows for forecasting."""
    X_windows = []
    y_windows = []
    discharge_ids = []
    end_times = []

    df = df.sort_values(["discharge_ID","time"])

    for discharge_id, shot in df.groupby("discharge_ID"):
        signals = shot[features].to_numpy(dtype = np.float32)
        labels = shot["density_limit_phase"].to_numpy(dtype = np.float32)
        times = shot["time"].to_numpy()

        final_end = len(shot) - forecast_horizon

        for end in range (window_size, final_end+1):
            start = end - window_size
            input_window = signals[start:end]
            input_labels = labels[start:end]
            future_labels = labels[end:end + forecast_horizon]

            if input_labels[-1] == 1:
                continue

            target = int(np.any(future_labels ==1))
            X_windows.append(input_window)
            y_windows.append(target)
            discharge_ids.append(discharge_id)
            end_times.append([times[end-1]])

    return(
        np.asarray(X_windows),
        np.asarray(y_windows),
        np.asarray(discharge_ids),
        np.asarray(end_times))