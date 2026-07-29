#Converts each plasma discharge time series into sliding windows for forecasting.
#Each input contains 50 time steps of six plasma signals and its binary target.
# Indicates whether the density-limit phase occurs within the next 20 time steps.
# Windows remain within individual discharges, and already-unstable inputs are skipped.

# Returns:
#   X_windows: Array with shape (n_windows, 50, 6) containing the input signals.
#   y_windows: Array with shape (n_windows,) containing binary forecast targets.
#   discharge_ids: Array with shape (n_windows,) identifying each window's discharge.
#   end_times: Array containing the time of the final sample in each input window.

import numpy as np 
import pandas as pd

features = [
    "density",
    "plasma_current",
    "elongation",
    "minor_radius",
    "toroidal_B_field",
    "triangularity",
]

window_size = 50
forecast_horizon = 20 

def make_windows(df, window_size,forecast_horizon):
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
        np.asarray(end_times)
    )