"""Smoke tests for the end-to-end training and inference pipeline."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Make the package importable without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pumpitup.config import TARGET_COL
from pumpitup.data.synthetic import generate_synthetic_pump_data
from pumpitup.evaluation.metrics import compute_metrics
from pumpitup.features.preprocess import build_preprocess_pipeline
from pumpitup.models.predict import predict
from pumpitup.models.train import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_pump_data(n_samples=100, seed=0)


@pytest.fixture(scope="module")
def fitted_pipeline(synthetic_df):
    return train_model(synthetic_df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthetic_data_shape(synthetic_df):
    """Synthetic dataset has the expected shape and columns."""
    assert len(synthetic_df) == 100
    assert TARGET_COL in synthetic_df.columns


def test_synthetic_data_has_all_three_labels(synthetic_df):
    """All three pump status labels are present in the synthetic data."""
    labels = set(synthetic_df[TARGET_COL].unique())
    assert labels == {"functional", "non functional", "functional needs repair"}


def test_pipeline_fit_predict(synthetic_df, fitted_pipeline):
    """Pipeline can fit on synthetic data and produce predictions with correct shape."""
    X = synthetic_df.drop(columns=[TARGET_COL])
    predictions = predict(fitted_pipeline, X)
    assert len(predictions) == len(synthetic_df)
    assert set(predictions.unique()).issubset(
        {"functional", "non functional", "functional needs repair"}
    )


def test_preprocessing_handles_missing_values():
    """Preprocessing pipeline does not crash on columns with NaN values."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "amount_tsh": rng.uniform(0, 1000, n),
            "gps_height": rng.uniform(0, 2500, n),
            "basin": rng.choice(["A", "B", "C"], n),
        }
    )
    # Inject missing values
    df.loc[0, "amount_tsh"] = np.nan
    df.loc[1, "basin"] = np.nan

    preprocessor = build_preprocess_pipeline(
        numeric_features=["amount_tsh", "gps_height"],
        categorical_features=["basin"],
    )
    result = preprocessor.fit_transform(df)
    assert result.shape[0] == n
    assert not np.any(np.isnan(result))


def test_metrics_output():
    """compute_metrics returns accuracy and f1_macro within valid range."""
    y_true = pd.Series(["functional", "non functional", "functional needs repair"])
    y_pred = pd.Series(["functional", "non functional", "functional"])
    metrics = compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
