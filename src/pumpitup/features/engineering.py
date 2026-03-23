"""Feature engineering utilities for the Pump It Up dataset."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Tanzania geographic center (approximate)
_TANZANIA_CENTER_LAT = -6.369028
_TANZANIA_CENTER_LON = 34.888822

# Reference year used in the original dataset collection
_DATASET_YEAR = 2013


def add_pump_age(df: pd.DataFrame, reference_year: int = _DATASET_YEAR) -> pd.DataFrame:
    """Add ``pump_age`` and ``pump_age_category`` columns derived from ``construction_year``.

    Args:
        df: Input DataFrame. Must contain a ``construction_year`` column.
        reference_year: The year against which pump age is calculated.

    Returns:
        Copy of *df* with ``pump_age`` (int) and ``pump_age_category`` (categorical)
        columns appended.  If ``construction_year`` is absent, *df* is returned unchanged.
    """
    df_out = df.copy()
    if "construction_year" not in df_out.columns:
        return df_out

    df_out["pump_age"] = (reference_year - df_out["construction_year"]).clip(lower=0)

    df_out["pump_age_category"] = pd.cut(
        df_out["pump_age"],
        bins=[0, 5, 10, 20, float("inf")],
        labels=["new (0-5)", "young (5-10)", "mid (10-20)", "old (20+)"],
        right=True,
    )
    return df_out


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add recording-date features derived from a ``date_recorded`` column.

    New columns added (when ``date_recorded`` is present):

    * ``recorded_month`` — calendar month (1–12)
    * ``recorded_season`` — one of ``winter``, ``spring``, ``summer``, ``autumn``
    * ``recorded_day_of_year`` — day of year (1–366)

    Args:
        df: Input DataFrame. Optionally contains a ``date_recorded`` string/datetime
            column.

    Returns:
        Copy of *df* with new date-derived columns.  If ``date_recorded`` is absent,
        *df* is returned unchanged.
    """
    df_out = df.copy()
    if "date_recorded" not in df_out.columns:
        return df_out

    dates = pd.to_datetime(df_out["date_recorded"], errors="coerce")
    df_out["recorded_month"] = dates.dt.month
    df_out["recorded_day_of_year"] = dates.dt.dayofyear

    def _month_to_season(month: float) -> str | None:
        if pd.isna(month):
            return None
        m = int(month)
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

    df_out["recorded_season"] = df_out["recorded_month"].map(_month_to_season)
    return df_out


def _haversine_km_vectorized(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    """Vectorised Haversine distance (km) from a fixed reference point."""
    r = 6371.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    clat_r = math.radians(center_lat)
    clon_r = math.radians(center_lon)
    dlat = clat_r - lat_r
    dlon = clon_r - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * math.cos(clat_r) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_geographical_features(
    df: pd.DataFrame,
    center_lat: float = _TANZANIA_CENTER_LAT,
    center_lon: float = _TANZANIA_CENTER_LON,
) -> pd.DataFrame:
    """Add geographical features derived from GPS coordinates.

    New columns added (when ``latitude`` and ``longitude`` are present):

    * ``distance_from_center`` — Haversine distance (km) from *center_lat/lon*

    When ``gps_height`` is also present:

    * ``elevation_category`` — one of ``low``, ``mid``, ``high``, ``very high``

    Args:
        df: Input DataFrame.
        center_lat: Reference latitude (default: Tanzania geographic center).
        center_lon: Reference longitude (default: Tanzania geographic center).

    Returns:
        Copy of *df* with new geographic columns.
    """
    df_out = df.copy()
    if "latitude" in df_out.columns and "longitude" in df_out.columns:
        df_out["distance_from_center"] = _haversine_km_vectorized(
            df_out["latitude"].to_numpy(dtype=float),
            df_out["longitude"].to_numpy(dtype=float),
            center_lat,
            center_lon,
        )

    if "gps_height" in df_out.columns:
        df_out["elevation_category"] = pd.cut(
            df_out["gps_height"],
            bins=[-float("inf"), 500, 1000, 1500, float("inf")],
            labels=["low", "mid", "high", "very high"],
        )

    return df_out


def add_aggregation_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "status_group",
    failure_label: str = "non functional",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add group-level aggregate features derived from the training set.

    For each column in *group_cols* (that exists in *train_df*) the following
    features are computed on the training set and then mapped onto both train
    and test data:

    * ``<col>_pump_count`` — number of pumps in that group
    * ``<col>_avg_age``    — mean ``pump_age`` per group (if ``pump_age`` exists)
    * ``<col>_failure_rate`` — fraction of pumps labelled *failure_label* (if
      *target_col* exists)

    Aggregations are derived **only** from *train_df* to avoid data leakage.

    Args:
        train_df: Training DataFrame (may include *target_col*).
        test_df: Test/validation DataFrame (target column should **not** be present,
            but the function is safe even if it is).
        group_cols: Categorical columns to group by.
        target_col: Name of the target column used for failure-rate computation.
        failure_label: Target label considered a failure (default: ``"non functional"``).

    Returns:
        Tuple of ``(train_with_agg, test_with_agg)`` DataFrames.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    for col in group_cols:
        if col not in train_out.columns:
            continue

        # Pump count
        counts = train_out[col].value_counts().to_dict()
        train_out[f"{col}_pump_count"] = train_out[col].map(counts)
        test_out[f"{col}_pump_count"] = test_out[col].map(counts)

        # Average pump age
        if "pump_age" in train_out.columns:
            age_mean = train_out.groupby(col)["pump_age"].mean().to_dict()
            train_out[f"{col}_avg_age"] = train_out[col].map(age_mean)
            test_out[f"{col}_avg_age"] = test_out[col].map(age_mean)

        # Failure rate
        if target_col in train_out.columns:
            failure_rate = (
                train_out.groupby(col)[target_col]
                .apply(lambda s: (s == failure_label).sum() / len(s))
                .to_dict()
            )
            train_out[f"{col}_failure_rate"] = train_out[col].map(failure_rate)
            test_out[f"{col}_failure_rate"] = test_out[col].map(failure_rate)

    return train_out, test_out
