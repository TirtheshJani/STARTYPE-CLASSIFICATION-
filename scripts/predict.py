"""Run the Star Type classifier against a CSV of stars.

Example
-------
    python -m scripts.predict --model models/star_classifier.joblib --input new_stars.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import normalize_color  # noqa: E402
from src.star_classifier import load_model, predict_dataframe  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict star types.")
    parser.add_argument("--model", required=True, help="Path to a saved joblib pipeline")
    parser.add_argument("--input", required=True, help="CSV of stars to classify")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write predictions as CSV (defaults to stdout)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = load_model(args.model)
    df = pd.read_csv(args.input)
    if "Color" in df.columns:
        df["Color"] = df["Color"].map(normalize_color)
    if "Spectral_Class" in df.columns:
        df["Spectral_Class"] = df["Spectral_Class"].str.strip().str.upper()
    predictions = predict_dataframe(pipeline, df)
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}")
    else:
        print(predictions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
