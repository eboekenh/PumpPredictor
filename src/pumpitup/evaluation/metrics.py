"""Evaluation metrics for multiclass classification."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline


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


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate *pipeline* with stratified k-fold cross-validation.

    Args:
        pipeline: An unfitted (or freshly cloned) sklearn ``Pipeline``.
        X: Feature DataFrame.
        y: Target Series.
        cv: Number of folds.
        random_state: Seed for the ``StratifiedKFold`` splitter.

    Returns:
        Dictionary with keys ``accuracy_mean``, ``accuracy_std``,
        ``f1_macro_mean``, and ``f1_macro_std`` (all floats).
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    cv_results = sk_cross_validate(pipeline, X, y, cv=kf, scoring=scoring)
    return {
        "accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "accuracy_std": float(cv_results["test_accuracy"].std()),
        "f1_macro_mean": float(cv_results["test_f1_macro"].mean()),
        "f1_macro_std": float(cv_results["test_f1_macro"].std()),
    }


def compare_models(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Compare multiple fitted models on the same held-out dataset.

    Args:
        models: Mapping of ``{model_name: fitted_pipeline_or_classifier}``.
        X: Feature DataFrame (not seen during training).
        y: Ground-truth labels.

    Returns:
        DataFrame with columns ``model``, ``accuracy``, and ``f1_macro``,
        sorted by ``accuracy`` descending.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X)
        metrics = compute_metrics(y, pd.Series(y_pred, index=y.index))
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
