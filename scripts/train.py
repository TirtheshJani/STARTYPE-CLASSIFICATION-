"""Train the Star Type classifier from the command line.

Example
-------
    python -m scripts.train --dataset Stars.csv --model models/star_classifier.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.star_classifier import train  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Star Type classifier.")
    parser.add_argument("--dataset", default="Stars.csv", help="Path to Stars.csv")
    parser.add_argument(
        "--model",
        default="models/star_classifier.joblib",
        help="Where to save the fitted pipeline",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, report = train(
        dataset_path=args.dataset,
        model_path=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(report.pretty())
    print(f"\nModel saved to: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
