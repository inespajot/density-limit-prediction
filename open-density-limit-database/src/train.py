import json 
import pickle
from pathlib import Path

import numpy as np 
import pandas as pd
from tensorflow import keras

from src.preprocessing import (features, split_by_discharge,scale_features,make_windows)
from src.models import build_cnn

DATA_PATH = Path("../DL_DataFrame.csv")
OUTPUT_DIR = Path("../outputs")

window_size = 50
forecast_horizon = 20
test_size = 0.20
val_size = 0.20
random_state = 42

def main():
    
    keras.utils.set_random_seed(random_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    train_df, val_df, test_df = split_by_discharge(df,random_state=random_state)
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df)

