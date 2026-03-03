"""Project-wide constants and default settings."""

import os

# Reproducibility
RANDOM_SEED = 42

# Target column name (matches DrivenData competition)
TARGET_COL = "status_group"

# Default artifact paths
ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
DEFAULT_PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, "predictions.csv")
