"""Matplotlib helpers for exploring the Stars dataset and model outputs."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_utils import STAR_TYPE_NAMES


def plot_hr_diagram(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot a Hertzsprung--Russell style diagram (temperature vs luminosity)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    labels = df["Type"].map(STAR_TYPE_NAMES)
    sns.scatterplot(
        data=df.assign(Label=labels),
        x="Temperature",
        y="L",
        hue="Label",
        palette="tab10",
        alpha=0.85,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Temperature (K, log)")
    ax.set_ylabel("Luminosity (L / L$_\\odot$, log)")
    ax.set_title("Hertzsprung\u2013Russell Diagram")
    ax.legend(title="Star Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax


def plot_confusion(matrix: np.ndarray, class_labels: Iterable[str], ax: plt.Axes | None = None) -> plt.Axes:
    """Render a confusion matrix as an annotated heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_labels),
        yticklabels=list(class_labels),
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax
