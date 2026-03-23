"""Tests for visualization utilities and evaluation helpers."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Use the non-interactive matplotlib backend to avoid display requirements
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pumpitup.evaluation.metrics import compare_models, cross_validate_pipeline
from pumpitup.models.train import get_feature_importances, train_model
from pumpitup.data.synthetic import generate_synthetic_pump_data
from pumpitup.config import TARGET_COL
from pumpitup.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_target_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_pump_data(n_samples=150, seed=7)


@pytest.fixture(scope="module")
def fitted_pipeline(synthetic_df):
    return train_model(synthetic_df)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def test_cross_validate_pipeline_keys(synthetic_df):
    """cross_validate_pipeline returns the four expected keys."""
    pipeline = train_model(synthetic_df)
    X = synthetic_df.drop(columns=[TARGET_COL])
    y = synthetic_df[TARGET_COL]
    result = cross_validate_pipeline(pipeline, X, y, cv=3)
    assert set(result.keys()) == {"accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std"}


def test_cross_validate_pipeline_values(synthetic_df):
    """All cross-validation metrics must be in [0, 1]."""
    pipeline = train_model(synthetic_df)
    X = synthetic_df.drop(columns=[TARGET_COL])
    y = synthetic_df[TARGET_COL]
    result = cross_validate_pipeline(pipeline, X, y, cv=3)
    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_compare_models_returns_dataframe(synthetic_df, fitted_pipeline):
    """compare_models returns a DataFrame with expected columns."""
    X = synthetic_df.drop(columns=[TARGET_COL])
    y = synthetic_df[TARGET_COL]
    results = compare_models({"RF": fitted_pipeline}, X, y)
    assert isinstance(results, pd.DataFrame)
    assert "model" in results.columns
    assert "accuracy" in results.columns
    assert "f1_macro" in results.columns


def test_compare_models_multiple(synthetic_df):
    """compare_models handles more than one model and sorts by accuracy."""
    pipeline_a = train_model(synthetic_df, random_state=1)
    pipeline_b = train_model(synthetic_df, random_state=2)
    X = synthetic_df.drop(columns=[TARGET_COL])
    y = synthetic_df[TARGET_COL]
    results = compare_models({"ModelA": pipeline_a, "ModelB": pipeline_b}, X, y)
    # Results must be sorted descending by accuracy
    assert results["accuracy"].iloc[0] >= results["accuracy"].iloc[1]


def test_get_feature_importances(fitted_pipeline):
    """get_feature_importances returns a non-empty DataFrame."""
    df = get_feature_importances(fitted_pipeline)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "feature" in df.columns
    assert "importance" in df.columns
    # Must be sorted descending
    assert (df["importance"].diff().dropna() <= 0).all()


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------


def test_plot_target_distribution_returns_figure(synthetic_df):
    fig = plot_target_distribution(synthetic_df, target_col=TARGET_COL)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_feature_importance_returns_figure():
    names = ["feat_a", "feat_b", "feat_c"]
    importances = np.array([0.5, 0.3, 0.2])
    fig = plot_feature_importance(names, importances, top_n=3)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_confusion_matrix_returns_figure():
    y_true = pd.Series(["functional", "non functional", "functional needs repair", "functional"])
    y_pred = pd.Series(["functional", "functional", "functional needs repair", "functional"])
    fig = plot_confusion_matrix(y_true, y_pred)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_model_comparison_returns_figure():
    results = pd.DataFrame(
        {
            "model": ["RF", "XGB"],
            "accuracy": [0.80, 0.82],
            "f1_macro": [0.73, 0.75],
        }
    )
    fig = plot_model_comparison(results, metrics=["accuracy", "f1_macro"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")
