"""Loading and preprocessing utilities for the Stars dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

NUMERIC_FEATURES: list[str] = ["Temperature", "L", "R", "A_M"]
CATEGORICAL_FEATURES: list[str] = ["Color", "Spectral_Class"]
TARGET: str = "Type"

STAR_TYPE_NAMES: dict[int, str] = {
    0: "Red Dwarf",
    1: "Brown Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Super Giant",
    5: "Hyper Giant",
}

# Canonical color labels used to normalise the free-form Color column.
_COLOR_ALIASES: dict[str, str] = {
    "blue white": "blue-white",
    "blue-white": "blue-white",
    "bluewhite": "blue-white",
    "blue": "blue",
    "red": "red",
    "white": "white",
    "whitish": "white",
    "yellowish": "yellow",
    "yellow": "yellow",
    "yellowish white": "yellow-white",
    "yellow-white": "yellow-white",
    "white-yellow": "yellow-white",
    "white yellow": "yellow-white",
    "orange": "orange",
    "orange-red": "orange-red",
    "pale yellow orange": "yellow-orange",
}


def normalize_color(value: str) -> str:
    """Return a canonical, lower-case color label.

    The raw Stars.csv uses inconsistent casing and punctuation for the
    ``Color`` column (e.g. ``"Blue White"`` vs ``"Blue-white"``). This
    helper collapses those variants so the one-hot encoder does not
    treat them as distinct categories.
    """
    if not isinstance(value, str):
        return str(value)
    key = " ".join(value.strip().lower().split())
    return _COLOR_ALIASES.get(key, key)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the Stars CSV and normalise the categorical columns."""
    df = pd.read_csv(path)
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Stars dataset is missing columns: {sorted(missing)}")
    df["Color"] = df["Color"].map(normalize_color)
    df["Spectral_Class"] = df["Spectral_Class"].str.strip().str.upper()
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` using the standard feature / target layout."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[features].copy(), df[TARGET].copy()
