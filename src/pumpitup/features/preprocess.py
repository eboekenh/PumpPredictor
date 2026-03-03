"""Feature preprocessing pipeline using ColumnTransformer."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocess_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes and encodes features.

    Numeric pipeline:
        - ``SimpleImputer(strategy="median")``

    Categorical pipeline:
        - ``SimpleImputer(strategy="most_frequent")``
        - ``OneHotEncoder(handle_unknown="ignore", sparse_output=False)``

    Args:
        numeric_features: Column names to treat as numeric.
        categorical_features: Column names to treat as categorical.

    Returns:
        Unfitted ``ColumnTransformer`` ready to be embedded in a ``Pipeline``.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
