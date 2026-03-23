"""Model training utilities."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from pumpitup.config import RANDOM_SEED, TARGET_COL
from pumpitup.features.preprocess import build_preprocess_pipeline

# Default numeric and categorical feature columns matching the synthetic schema.
# When using real data these can be overridden by the caller.
DEFAULT_NUMERIC_FEATURES = [
    "amount_tsh",
    "gps_height",
    "longitude",
    "latitude",
    "num_private",
    "population",
    "construction_year",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "basin",
    "extraction_type",
    "water_quality",
]


def train_model(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Fit a full sklearn Pipeline (preprocessing + classifier) on *df*.

    Args:
        df: Training DataFrame including the target column.
        target_col: Name of the target column.
        numeric_features: Numeric feature column names (defaults to
            ``DEFAULT_NUMERIC_FEATURES``).
        categorical_features: Categorical feature column names (defaults to
            ``DEFAULT_CATEGORICAL_FEATURES``).
        random_state: Random seed for the classifier.

    Returns:
        Fitted ``sklearn.pipeline.Pipeline``.
    """
    if numeric_features is None:
        numeric_features = [c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns]
    if categorical_features is None:
        categorical_features = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in df.columns]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_preprocess_pipeline(numeric_features, categorical_features)

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipeline.fit(X, y)
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> None:
    """Persist a fitted pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path: Destination file path (e.g. ``artifacts/model.joblib``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(pipeline, path)


def get_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature importances from a fitted pipeline.

    The pipeline must have a ``preprocessor`` step (a ``ColumnTransformer``)
    and a ``classifier`` step that exposes ``feature_importances_``.

    Args:
        pipeline: Fitted sklearn ``Pipeline`` produced by :func:`train_model`.

    Returns:
        DataFrame with columns ``feature`` and ``importance``, sorted by
        importance descending.

    Raises:
        AttributeError: If the classifier does not have ``feature_importances_``.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Collect names from each transformer in the ColumnTransformer
    feature_names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(cols).tolist())
        else:
            feature_names.extend(cols)

    importances = classifier.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
