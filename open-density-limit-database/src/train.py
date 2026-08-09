import json
import pickle
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from tensorflow import keras

from src.preprocessing import features, split_by_discharge, scale_features, make_windows
from src.models import build_model, compile_model
from src.evaluate import select_threshold, evaluate_at_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "DL_DataFrame.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

window_size = 50
forecast_horizon = 20
test_size = 0.20
val_size = 0.20
random_state = 42
window_stride = 1

model_name = "cnn"

training_configs = {
    "cnn": {
        "epochs": 20,
        "batch_size": 64,
        "patience": 15,
        "learning_rate": 1e-3,
        "classification_threshold": 0.5,
        "minimum_recall": 0.95,
    },
    "tcn": {
        "epochs": 200,
        "batch_size": 64,
        "patience": 20,
        "learning_rate": 1e-3,
        "classification_threshold": 0.5,
        "minimum_recall": 0.95,
    },
    "gru": {
        "epochs": 200,
        "batch_size": 32,
        "patience": 20,
        "learning_rate": 1e-3,
        "classification_threshold": 0.5,
        "minimum_recall": 0.95,
    },
}


def train(experiment_dir, config):
    keras.utils.set_random_seed(random_state)

    df = pd.read_csv(DATA_PATH)

    train_df, val_df, test_df = split_by_discharge(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )
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

    print("Model", model_name)
    print("Training windows:", X_train.shape)
    print("Validation windows:", X_val.shape)
    print("Test windows:", X_test.shape)

    print("Training positives:", y_train.sum())
    print("Validation positives:", y_val.sum())
    print("Test positives: ", y_test.sum())

    if len(X_train) == 0:
        raise ValueError("No training windows were generated.")

    if len(X_val) == 0:
        raise ValueError("No validation windows were generated.")

    if len(X_test) == 0:
        raise ValueError("No test windows were generated.")

    if (
        np.unique(y_train).size < 2
        or np.unique(y_val).size < 2
        or np.unique(y_test).size < 2
    ):
        raise ValueError(
            "Test windows must contain both positive and negative targets."
        )

    model = build_model(
        name=model_name, window_size=window_size, feature_number=len(features)
    )

    model = compile_model(
        model,
        learning_rate=config["learning_rate"],
        classification_threshold=config["classification_threshold"],
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=config["patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            str(experiment_dir / "best_model.keras"),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(
            str(experiment_dir / "training_history.csv"),
        ),
    ]

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)

    class_weight = {0: 1.0, 1: negative_count / max(positive_count, 1)}
    print("Class weights:", class_weight)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        class_weight=class_weight,
    )

    val_probabilities = model.predict(X_val).ravel()
    threshold_selection = select_threshold(
        y_val, val_probabilities, minimum_recall=config["minimum_recall"]
    )

    selected_threshold = threshold_selection["threshold"]

    print(
        "Validation threshold selection:",
        threshold_selection,
    )

    keras_metrics = model.evaluate(X_test, y_test, return_dict=True)

    keras_metrics = {name: float(value) for name, value in keras_metrics.items()}

    test_probabilities = model.predict(X_test).ravel()

    threshold_metrics, test_predictions = evaluate_at_threshold(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    metrics = {
        "validation_threshold_selection": threshold_selection,
        "keras_test_metrics": keras_metrics,
        "threshold_test_metrics": threshold_metrics,
    }

    print("Test Metrics:", metrics)

    model.save(str(experiment_dir / "final_model.keras"))

    with open(experiment_dir / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    with open(experiment_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    experiment_config = {
        "model_name": model_name,
        "window_size": window_size,
        "forecast_horizon": forecast_horizon,
        "window_stride": window_stride,
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "features": features,
        "training": config,
        "selected_threshold": selected_threshold,
    }

    with open(
        experiment_dir / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(experiment_config, file, indent=2)

    prediction_df = pd.DataFrame(
        {
            "discharge_ID": test_discharge_ids,
            "time": test_end_times,
            "target": y_test,
            "probability": test_probabilities,
            "prediction": test_predictions,
        }
    )

    prediction_df.to_csv(
        experiment_dir / "test_predictions.csv",
        index=False,
    )

    return history, metrics


def main():
    config = training_configs[model_name]

    experiment_dir = OUTPUT_DIR / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "training_log.txt"
    with open(log_path, "w") as log_file:
        with redirect_stdout(log_file):
            train(experiment_dir, config)


if __name__ == "__main__":
    main()
