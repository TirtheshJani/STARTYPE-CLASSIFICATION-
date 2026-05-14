"""Generate the figures embedded in README.md.

Running this script regenerates every image under ``docs/images/`` from
``Stars.csv``, training the same pipeline used in :mod:`scripts.train`.

Usage
-----
    python -m scripts.generate_report_assets
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    STAR_TYPE_NAMES,
    load_dataset,
    split_features_target,
)
from src.star_classifier import build_pipeline  # noqa: E402
from src.visualize import plot_confusion, plot_hr_diagram  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def hr_diagram(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_hr_diagram(df, ax=ax)
    return _save(fig, "hr_diagram.png")


def class_distribution(df: pd.DataFrame) -> Path:
    counts = df["Type"].map(STAR_TYPE_NAMES).value_counts().reindex(
        list(STAR_TYPE_NAMES.values())
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        hue=counts.index,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for i, v in enumerate(counts.values):
        ax.text(v + 0.4, i, str(v), va="center", fontsize=11)
    ax.set_xlabel("Number of stars")
    ax.set_ylabel("")
    ax.set_title("Class distribution (240 stars, 40 per class)")
    ax.set_xlim(0, counts.max() + 6)
    sns.despine()
    return _save(fig, "class_distribution.png")


def feature_correlation(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr = df[NUMERIC_FEATURES + ["Type"]].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Pearson correlation of numeric features with Type")
    return _save(fig, "feature_correlation.png")


def pairplot(df: pd.DataFrame) -> Path:
    labeled = df.assign(Label=df["Type"].map(STAR_TYPE_NAMES))
    grid = sns.pairplot(
        labeled,
        vars=NUMERIC_FEATURES,
        hue="Label",
        palette="tab10",
        plot_kws={"alpha": 0.7, "s": 22},
        diag_kind="kde",
        height=2.0,
    )
    grid.figure.suptitle("Pairwise relationships of numeric features", y=1.02)
    path = OUT_DIR / "pairplot.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(grid.figure)
    return path


def confusion_and_importance(df: pd.DataFrame) -> tuple[Path, Path]:
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline(random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    class_labels = [STAR_TYPE_NAMES[i] for i in sorted(y.unique())]
    plot_confusion(cm, class_labels, ax=ax)
    cm_path = _save(fig, "confusion_matrix.png")

    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = list(NUMERIC_FEATURES) + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )
    importances = classifier.feature_importances_
    order = np.argsort(importances)[::-1][:12]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=importances[order],
        y=[feature_names[i] for i in order],
        hue=[feature_names[i] for i in order],
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Random Forest feature importance")
    ax.set_ylabel("")
    ax.set_title("Top features driving the prediction")
    sns.despine()
    fi_path = _save(fig, "feature_importance.png")
    return cm_path, fi_path


def main() -> int:
    df = load_dataset(Path(__file__).resolve().parents[1] / "Stars.csv")
    sns.set_theme(style="whitegrid")
    written = [
        hr_diagram(df),
        class_distribution(df),
        feature_correlation(df),
        pairplot(df),
        *confusion_and_importance(df),
    ]
    for path in written:
        print(f"wrote {path.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
