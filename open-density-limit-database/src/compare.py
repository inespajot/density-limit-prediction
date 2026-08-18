import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "experiment_summary.csv"


def load_experiment(experiment_dir):
    """Return one flat summary row for a completed experiment."""
    config_path = experiment_dir / "experiment_config.json"
    metrics_path = experiment_dir / "metrics.json"

    if not config_path.exists() or not metrics_path.exists():
        return None

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    training = config["training"]
    data = config["data"]
    validation = metrics["keras_validation_metrics"]
    threshold = metrics["validation_threshold_selection"]
    test = metrics.get("keras_test_metrics", {})
    test_threshold = metrics.get("threshold_test_metrics", {})

    return {
        "run_name": config["run_name"],
        "model": config["model"],
        "seed": config.get("random_state", 42),
        "window_size": data["window_size"],
        "forecast_horizon": data["forecast_horizon"],
        "window_stride": data["window_stride"],
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "val_pr_auc": validation.get("pr_auc"),
        "val_roc_auc": validation.get("roc_auc"),
        "val_precision_at_selected_threshold": threshold.get("precision"),
        "val_recall_at_selected_threshold": threshold.get("recall"),
        "test_pr_auc": test.get("pr_auc"),
        "test_roc_auc": test.get("roc_auc"),
        "test_precision_at_selected_threshold": test_threshold.get("precision"),
        "test_recall_at_selected_threshold": test_threshold.get("recall"),
    }


def main():
    if not EXPERIMENTS_DIR.exists():
        print("No experiments directory exists yet.")
        return

    rows = [
        row
        for directory in sorted(EXPERIMENTS_DIR.iterdir())
        if directory.is_dir()
        for row in [load_experiment(directory)]
        if row is not None
    ]

    if not rows:
        print("No completed experiments found.")
        return

    summary = pd.DataFrame(rows).sort_values(
        "val_pr_auc", ascending=False, na_position="last"
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    display_columns = [
        "run_name",
        "model",
        "window_stride",
        "learning_rate",
        "val_pr_auc",
        "val_recall_at_selected_threshold",
        "test_pr_auc",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nSaved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
