"""Generate a small synthetic dataset that mimics the Pump It Up schema."""

import numpy as np
import pandas as pd

from pumpitup.config import RANDOM_SEED, TARGET_COL

# Plausible categorical values
_BASIN_VALUES = ["Lake Victoria", "Pangani", "Rufiji", "Internal", "Ruvuma / Southern Coast"]
_EXTRACTION_TYPES = ["gravity", "handpump", "submersible", "motorpump", "other"]
_WATER_QUALITY = ["soft", "salty", "milky", "coloured", "fluoride", "unknown"]
_STATUS_LABELS = ["functional", "non functional", "functional needs repair"]


def generate_synthetic_pump_data(n_samples: int = 200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic DataFrame with a schema similar to the Pump It Up dataset.

    Includes numeric and categorical features plus a multiclass target column.

    Args:
        n_samples: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with mixed feature types and a ``status_group`` target column.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            # Numeric features
            "amount_tsh": rng.uniform(0, 1000, n_samples),
            "gps_height": rng.uniform(-90, 2500, n_samples),
            "longitude": rng.uniform(29.0, 40.5, n_samples),
            "latitude": rng.uniform(-11.5, -1.0, n_samples),
            "num_private": rng.integers(0, 10, n_samples),
            "population": rng.integers(0, 1000, n_samples),
            "construction_year": rng.integers(1960, 2014, n_samples),
            # Categorical features
            "basin": rng.choice(_BASIN_VALUES, n_samples),
            "extraction_type": rng.choice(_EXTRACTION_TYPES, n_samples),
            "water_quality": rng.choice(_WATER_QUALITY, n_samples),
            # Target
            TARGET_COL: rng.choice(_STATUS_LABELS, n_samples),
        }
    )

    # Introduce a small fraction of missing values to exercise imputers
    for col in ["amount_tsh", "gps_height", "construction_year", "water_quality"]:
        mask = rng.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan

    return df
