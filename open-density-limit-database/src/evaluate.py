import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def _validate_inputs(y_true, probabilities):
    """Validate target labels and predicted probabilities."""

    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities).ravel()

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Targets and probabilities must have the same length. "
            f"Received {len(y_true)} targets and "
            f"{len(probabilities)} probabilities."
        )

    if len(y_true) == 0:
        raise ValueError("Targets cannot be empty.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities contain NaN.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    unique_targets = np.unique(y_true)

    if not np.all(np.isin(unique_targets, [0, 1])):
        raise ValueError("Targets must contain binary values only.")

    return y_true, probabilities


def select_threshold(y_true, probabilities, minimum_recall):
    """Select the most precise probability threshold that achieves high minimum recall."""

    y_true, probabilities = _validate_inputs(y_true, probabilities)

    if np.unique(y_true).size < 2:
        raise ValueError("Validation data must contain both classes.")

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    precision = precision[:-1]
    recall = recall[:-1]

    acceptable_indices = np.where(recall >= minimum_recall)[0]

    if len(acceptable_indices) == 0:
        raise ValueError(f"No threshold achieved {minimum_recall:.0%} recall.")

    acceptable_precision = precision[acceptable_indices]
    highest_precision = np.max(acceptable_precision)

    best_candidates = acceptable_indices[
        np.isclose(acceptable_precision, highest_precision)
    ]

    best_index = best_candidates[np.argmax(thresholds[best_candidates])]

    selected_precision = precision[best_index]
    selected_recall = recall[best_index]

    f1 = (2 * selected_precision * selected_recall) / (
        selected_precision + selected_recall + 1e-8
    )

    return {
        "threshold": float(thresholds[best_index]),
        "minimum_required_recall": float(minimum_recall),
        "precision": float(selected_precision),
        "recall": float(selected_recall),
        "f1": float(f1),
    }


def evaluate_at_threshold(y_true, probabilities, threshold):
    """Calculate test metrics using pre-selected threshold."""

    y_true, probabilities = _validate_inputs(y_true, probabilities)

    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": matrix.tolist(),
    }

    return metrics, predictions
