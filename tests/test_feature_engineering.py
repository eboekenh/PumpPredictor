"""Tests for feature engineering utilities."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pumpitup.features.engineering import (
    add_aggregation_features,
    add_date_features,
    add_geographical_features,
    add_pump_age,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_df():
    """Minimal DataFrame with the most common columns."""
    return pd.DataFrame(
        {
            "construction_year": [1990, 2005, 2010, 0, 2013],
            "date_recorded": ["2013-03-01", "2013-06-15", "2013-09-20", None, "2013-12-31"],
            "latitude": [-6.0, -8.5, -4.0, -11.0, -1.5],
            "longitude": [35.0, 33.0, 38.0, 30.0, 40.0],
            "gps_height": [200, 700, 1200, 1700, -50],
            "basin": ["A", "B", "A", "C", "B"],
            "status_group": [
                "functional",
                "non functional",
                "functional",
                "non functional",
                "functional needs repair",
            ],
        }
    )


# ---------------------------------------------------------------------------
# add_pump_age
# ---------------------------------------------------------------------------


def test_add_pump_age_creates_columns(base_df):
    out = add_pump_age(base_df)
    assert "pump_age" in out.columns
    assert "pump_age_category" in out.columns


def test_add_pump_age_values(base_df):
    out = add_pump_age(base_df, reference_year=2013)
    # construction_year=1990 → age=23
    assert out.loc[0, "pump_age"] == 23
    # construction_year=2013 → age=0
    assert out.loc[4, "pump_age"] == 0


def test_add_pump_age_no_negative(base_df):
    """pump_age must never be negative even for year=0."""
    out = add_pump_age(base_df)
    assert (out["pump_age"] >= 0).all()


def test_add_pump_age_missing_column():
    """Returns unchanged DataFrame when construction_year is absent."""
    df = pd.DataFrame({"latitude": [-5.0]})
    out = add_pump_age(df)
    assert "pump_age" not in out.columns


def test_add_pump_age_does_not_mutate(base_df):
    """Original DataFrame must not be modified."""
    original_cols = list(base_df.columns)
    _ = add_pump_age(base_df)
    assert list(base_df.columns) == original_cols


# ---------------------------------------------------------------------------
# add_date_features
# ---------------------------------------------------------------------------


def test_add_date_features_creates_columns(base_df):
    out = add_date_features(base_df)
    assert "recorded_month" in out.columns
    assert "recorded_season" in out.columns
    assert "recorded_day_of_year" in out.columns


def test_add_date_features_seasons(base_df):
    out = add_date_features(base_df)
    # March → spring
    assert out.loc[0, "recorded_season"] == "spring"
    # June → summer
    assert out.loc[1, "recorded_season"] == "summer"
    # September → autumn
    assert out.loc[2, "recorded_season"] == "autumn"
    # December → winter
    assert out.loc[4, "recorded_season"] == "winter"


def test_add_date_features_missing_column():
    df = pd.DataFrame({"latitude": [-5.0]})
    out = add_date_features(df)
    assert "recorded_month" not in out.columns


# ---------------------------------------------------------------------------
# add_geographical_features
# ---------------------------------------------------------------------------


def test_add_geographical_features_distance(base_df):
    out = add_geographical_features(base_df)
    assert "distance_from_center" in out.columns
    # All distances should be positive
    assert (out["distance_from_center"] >= 0).all()


def test_add_geographical_features_elevation(base_df):
    out = add_geographical_features(base_df)
    assert "elevation_category" in out.columns


def test_add_geographical_features_no_lat_lon():
    df = pd.DataFrame({"gps_height": [500]})
    out = add_geographical_features(df)
    assert "distance_from_center" not in out.columns
    # elevation_category should still be created
    assert "elevation_category" in out.columns


# ---------------------------------------------------------------------------
# add_aggregation_features
# ---------------------------------------------------------------------------


def test_add_aggregation_features_pump_count(base_df):
    train, test = add_aggregation_features(base_df, base_df.copy(), group_cols=["basin"])
    assert "basin_pump_count" in train.columns
    assert "basin_pump_count" in test.columns


def test_add_aggregation_features_failure_rate(base_df):
    with_age = add_pump_age(base_df)
    train, test = add_aggregation_features(
        with_age, with_age.copy(), group_cols=["basin"], target_col="status_group"
    )
    assert "basin_failure_rate" in train.columns
    # Failure rate must be in [0, 1]
    assert (train["basin_failure_rate"].dropna() >= 0).all()
    assert (train["basin_failure_rate"].dropna() <= 1).all()


def test_add_aggregation_features_avg_age(base_df):
    with_age = add_pump_age(base_df)
    train, test = add_aggregation_features(with_age, with_age.copy(), group_cols=["basin"])
    assert "basin_avg_age" in train.columns


def test_add_aggregation_features_no_leakage():
    """Statistics on test set must come only from training data."""
    train = pd.DataFrame(
        {
            "basin": ["A", "A", "B"],
            "pump_age": [10, 20, 5],
            "status_group": ["functional", "non functional", "functional"],
        }
    )
    # Test has a group 'C' that was never seen in training
    test = pd.DataFrame({"basin": ["A", "C"], "pump_age": [15, 8]})
    train_out, test_out = add_aggregation_features(train, test, group_cols=["basin"])
    # Group 'C' was not in training → NaN expected
    assert pd.isna(test_out.loc[test_out["basin"] == "C", "basin_pump_count"].values[0])
