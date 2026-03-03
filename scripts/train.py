#!/usr/bin/env python
"""CLI entrypoint for training a pump-status classifier."""

import argparse
import os
import sys

# Ensure the src/ layout is importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pumpitup.config import DEFAULT_MODEL_PATH, TARGET_COL
from pumpitup.data.io import load_csv
from pumpitup.data.synthetic import generate_synthetic_pump_data
from pumpitup.evaluation.metrics import compute_metrics
from pumpitup.models.train import save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pump-status classifier.")
    parser.add_argument(
        "--train-csv",
        default=None,
        help="Path to training CSV. If omitted, a synthetic dataset is used.",
    )
    parser.add_argument(
        "--target",
        default=TARGET_COL,
        help=f"Name of the target column (default: {TARGET_COL}).",
    )
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=500,
        help="Number of synthetic samples to generate when no CSV is supplied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_csv:
        print(f"Loading training data from {args.train_csv} …")
        df = load_csv(args.train_csv)
    else:
        print(f"No --train-csv provided. Generating {args.n_synthetic} synthetic samples …")
        df = generate_synthetic_pump_data(n_samples=args.n_synthetic)

    print(f"Training data shape: {df.shape}")

    pipeline = train_model(df, target_col=args.target)

    # Compute in-sample metrics as a sanity check.
    X = df.drop(columns=[args.target])
    y = df[args.target]
    y_pred = pipeline.predict(X)
    metrics = compute_metrics(y, y_pred)
    print("Training metrics (in-sample):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    save_model(pipeline, args.model_output)
    print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    main()
