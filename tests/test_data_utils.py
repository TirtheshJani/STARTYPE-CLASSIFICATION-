"""Unit tests for :mod:`src.data_utils`."""

from __future__ import annotations

import io
import textwrap

import pytest

from src.data_utils import load_dataset, normalize_color, split_features_target


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Blue White", "blue-white"),
        ("blue-white", "blue-white"),
        ("Blue-white", "blue-white"),
        ("Whitish", "white"),
        ("yellowish", "yellow"),
        ("White-Yellow", "yellow-white"),
        ("Pale yellow orange", "yellow-orange"),
    ],
)
def test_normalize_color_aliases(raw: str, expected: str) -> None:
    assert normalize_color(raw) == expected


def test_load_dataset_and_split(tmp_path) -> None:
    csv = tmp_path / "stars.csv"
    csv.write_text(
        textwrap.dedent(
            """\
            Temperature,L,R,A_M,Color,Spectral_Class,Type
            3068,0.0024,0.17,16.12,Red,M,0
            25000,0.056,0.0084,10.58,Blue White,B,2
            39000,204000,10.6,-4.7,Blue,O,3
            """
        )
    )
    df = load_dataset(csv)
    assert list(df["Color"]) == ["red", "blue-white", "blue"]
    assert list(df["Spectral_Class"]) == ["M", "B", "O"]

    X, y = split_features_target(df)
    assert list(X.columns) == [
        "Temperature",
        "L",
        "R",
        "A_M",
        "Color",
        "Spectral_Class",
    ]
    assert y.tolist() == [0, 2, 3]


def test_load_dataset_rejects_missing_columns(tmp_path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("Temperature,L\n3000,0.01\n")
    with pytest.raises(ValueError):
        load_dataset(csv)
