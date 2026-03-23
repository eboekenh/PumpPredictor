"""Visualization utilities for Pump It Up EDA and model evaluation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "status_group",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the distribution of the target variable as a bar chart.

    Args:
        df: DataFrame containing the target column.
        target_col: Name of the target column.
        ax: Optional existing ``Axes`` to draw on.  A new figure is created if
            *ax* is ``None``.

    Returns:
        The ``Figure`` containing the plot.
    """
    counts = df[target_col].value_counts()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    bars = ax.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c", "#f39c12"])
    ax.set_title("Pump Status Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Status", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=15)

    # Annotate bars with counts and percentages
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({count / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a horizontal bar chart of feature importances.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores aligned with *feature_names*.
        top_n: Number of top features to display.
        ax: Optional existing ``Axes``.

    Returns:
        The ``Figure`` containing the plot.
    """
    feat_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, top_n // 2)))
    else:
        fig = ax.get_figure()

    ax.barh(range(len(feat_df)), feat_df["importance"].values, color="steelblue")
    ax.set_yticks(range(len(feat_df)))
    ax.set_yticklabels(feat_df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional class labels for the axes; inferred from *y_true* if
            not supplied.
        ax: Optional existing ``Axes``.

    Returns:
        The ``Figure`` containing the plot.
    """
    if labels is None:
        labels = sorted(set(np.asarray(y_true)))  # type: ignore[arg-type]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("Normalised Confusion Matrix", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def plot_model_comparison(
    results: pd.DataFrame,
    metrics: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a grouped bar chart comparing multiple models across several metrics.

    Args:
        results: DataFrame with a ``model`` column and one column per metric.
            Typically the output of
            :func:`pumpitup.evaluation.metrics.compare_models`.
        metrics: Metric column names to include.  Defaults to all numeric
            columns except ``model``.
        ax: Optional existing ``Axes``.

    Returns:
        The ``Figure`` containing the plot.
    """
    if metrics is None:
        metrics = [c for c in results.columns if c != "model" and pd.api.types.is_numeric_dtype(results[c])]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    x = np.arange(len(results))
    width = 0.8 / len(metrics)
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, results[metric], width, label=metric, color=colors[i % len(colors)])

    model_col = "model" if "model" in results.columns else results.columns[0]
    ax.set_xticks(x)
    ax.set_xticklabels(results[model_col], rotation=15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig
