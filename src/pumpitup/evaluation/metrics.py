"""Evaluation metrics for multiclass classification."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy and macro-averaged F1 score.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with keys ``accuracy`` and ``f1_macro``.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
