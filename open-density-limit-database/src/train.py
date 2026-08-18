import argparse
import json
import pickle
import re
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

from src.evaluate import evaluate_at_threshold, select_threshold
from src.models import build_model, compile_model
from src.preprocessing import features, make_windows, scale_features, split_by_discharge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "DL_DataFrame.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"
SUPPORTED_MODELS = {"cnn", "tcn", "gru"}


def load_config(config_path):
    """Load and validate an experiment configuration."""
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    missing = {"data", "training"} - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    run_name = config.get("run_name")
    valid_name = isinstance(run_name, str) and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name
    )
    if not valid_name:
        raise ValueError(
            "run_name must start with a letter or number and contain only "
            "letters, numbers, underscores, and hyphens."
        )

    model_name = config.get("model")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {sorted(SUPPORTED_MODELS)}")

    return config


def validate_arrays(X_train, y_train, X_val, y_val, X_test, y_test, run_test):
    """Check that each required split contains usable windows and labels."""
    arrays = [("training", X_train, y_train), ("validation", X_val, y_val)]
    if run_test:
        arrays.append(("test", X_test, y_test))

    for split_name, X, y in arrays:
        if len(X) == 0:
            raise ValueError(f"No {split_name} windows were generated.")
        if np.unique(y).size < 2:
            raise ValueError(f"{split_name.title()} windows must contain both classes.")


def train(experiment_dir, config):
    """Train and evaluate one configured experiment."""
    model_name = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    model_config = config.get("model_parameters", {})
    evaluation_config = config.get("evaluation", {})

    random_state = int(config.get("random_state", 42))
    window_size = int(data_config["window_size"])
    forecast_horizon = int(data_config["forecast_horizon"])
    window_stride = int(data_config["window_stride"])
    test_size = float(data_config["test_size"])
    val_size = float(data_config["val_size"])
    run_test = bool(evaluation_config.get("run_test", False))

    keras.utils.set_random_seed(random_state)

    df = pd.read_csv(DATA_PATH)
    train_df, val_df, test_df = split_by_discharge(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df)

    X_train, y_train, _, _ = make_windows(
        train_df, window_size, forecast_horizon, window_stride
    )
    X_val, y_val, _, _ = make_windows(
        val_df, window_size, forecast_horizon, window_stride
    )
    X_test, y_test, test_discharge_ids, test_end_times = make_windows(
        test_df, window_size, forecast_horizon, window_stride
    )

    validate_arrays(X_train, y_train, X_val, y_val, X_test, y_test, run_test)

    print("Run:", config["run_name"])
    print("Model:", model_name)
    print("Training windows:", X_train.shape)
    print("Validation windows:", X_val.shape)
    print("Test windows:", X_test.shape)
    print("Training positives:", int(y_train.sum()))
    print("Validation positives:", int(y_val.sum()))
    print("Test positives:", int(y_test.sum()))

    model = build_model(
        name=model_name,
        window_size=window_size,
        feature_number=len(features),
        model_parameters=model_config,
    )
    model = compile_model(
        model,
        learning_rate=float(training_config["learning_rate"]),
        classification_threshold=float(
            training_config.get("classification_threshold", 0.5)
        ),
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(training_config["patience"]),
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
        keras.callbacks.CSVLogger(str(experiment_dir / "training_history.csv")),
    ]

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)
    class_weight = {0: 1.0, 1: negative_count / max(positive_count, 1)}
    print("Class weights:", class_weight)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(training_config["epochs"]),
        batch_size=int(training_config["batch_size"]),
        callbacks=callbacks,
        class_weight=class_weight,
    )

    val_probabilities = model.predict(X_val).ravel()
    threshold_selection = select_threshold(
        y_val,
        val_probabilities,
        minimum_recall=float(training_config["minimum_recall"]),
    )
    selected_threshold = threshold_selection["threshold"]
    validation_keras_metrics = {
        name: float(value)
        for name, value in model.evaluate(X_val, y_val, return_dict=True).items()
    }
    metrics = {
        "validation_threshold_selection": threshold_selection,
        "keras_validation_metrics": validation_keras_metrics,
    }
    print("Validation metrics:", metrics)

    if run_test:
        test_keras_metrics = {
            name: float(value)
            for name, value in model.evaluate(X_test, y_test, return_dict=True).items()
        }
        test_probabilities = model.predict(X_test).ravel()
        threshold_metrics, test_predictions = evaluate_at_threshold(
            y_test, test_probabilities, selected_threshold
        )
        metrics["keras_test_metrics"] = test_keras_metrics
        metrics["threshold_test_metrics"] = threshold_metrics

        prediction_df = pd.DataFrame(
            {
                "discharge_ID": test_discharge_ids,
                "time": test_end_times,
                "target": y_test,
                "probability": test_probabilities,
                "prediction": test_predictions,
            }
        )
        prediction_df.to_csv(experiment_dir / "test_predictions.csv", index=False)
        print("Test metrics:", threshold_metrics)

    model.save(str(experiment_dir / "final_model.keras"))

    with open(experiment_dir / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    saved_config = {
        **config,
        "features": features,
        "selected_threshold": selected_threshold,
    }
    with open(experiment_dir / "experiment_config.json", "w", encoding="utf-8") as file:
        json.dump(saved_config, file, indent=2)

    return history, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a density-limit model from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_dir = EXPERIMENTS_DIR / config["run_name"]

    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment {config['run_name']!r} already exists at {experiment_dir}. "
            "Choose a new run_name to avoid overwriting results."
        )

    experiment_dir.mkdir(parents=True)
    with open(experiment_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    log_path = experiment_dir / "training_log.txt"
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file):
                train(experiment_dir, config)
    except Exception:
        print(f"Experiment failed. See {log_path}")
        raise

    print(f"Experiment completed: {config['run_name']}")
    print(f"Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
