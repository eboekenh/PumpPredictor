"""Model inference utilities."""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(path: str) -> Pipeline:
    """Load a persisted joblib model artifact.

    Args:
        path: Path to the ``.joblib`` file.

    Returns:
        Fitted sklearn Pipeline.
    """
    return joblib.load(path)


def predict(pipeline: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Generate predictions for *df* using the given pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        df: Feature DataFrame (target column should NOT be present).

    Returns:
        Series of predicted labels with the same index as *df*.
    """
    labels = pipeline.predict(df)
    return pd.Series(labels, index=df.index, name="status_group")
