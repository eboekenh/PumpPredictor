#!/usr/bin/env python
"""CLI entrypoint for generating pump-status predictions."""

import argparse
import os
import sys

# Ensure the src/ layout is importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pumpitup.config import DEFAULT_MODEL_PATH, DEFAULT_PREDICTIONS_PATH
from pumpitup.data.io import load_csv, save_predictions
from pumpitup.models.predict import load_model, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pump-status predictions.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model artifact (default: artifacts/model.joblib).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input feature CSV.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_PREDICTIONS_PATH,
        help="Where to write predictions CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.model} …")
    pipeline = load_model(args.model)

    print(f"Loading input data from {args.input} …")
    df = load_csv(args.input)
    print(f"Input data shape: {df.shape}")

    predictions = predict(pipeline, df)
    save_predictions(predictions, args.output)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
