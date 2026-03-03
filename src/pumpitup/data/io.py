"""Data loading and saving utilities."""

import os
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def save_predictions(predictions: pd.Series, output_path: str) -> None:
    """Save predictions to a CSV file.

    Args:
        predictions: Series of predicted labels (index = row id).
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    predictions.to_csv(output_path, header=True)
