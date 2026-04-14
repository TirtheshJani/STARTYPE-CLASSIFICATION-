"""Scikit-learn pipeline for the Star Type classification task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    STAR_TYPE_NAMES,
    load_dataset,
    split_features_target,
)


@dataclass
class EvaluationReport:
    """Container for evaluation results produced by :func:`train`."""

    accuracy: float
    cv_mean: float
    cv_std: float
    confusion: np.ndarray
    report: str

    def pretty(self) -> str:
        return (
            f"Test accuracy : {self.accuracy:.4f}\n"
            f"5-fold CV     : {self.cv_mean:.4f} \u00b1 {self.cv_std:.4f}\n\n"
            f"{self.report}"
        )


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Return an unfitted preprocessing + Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def train(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, EvaluationReport]:
    """Train the pipeline on the Stars dataset and return evaluation metrics.

    Parameters
    ----------
    dataset_path:
        Path to ``Stars.csv``.
    model_path:
        Optional destination for the fitted pipeline (``joblib`` format).
    test_size:
        Fraction of the dataset held out for evaluation.
    random_state:
        Seed propagated to the splitter and classifier.
    """
    df = load_dataset(dataset_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    report = EvaluationReport(
        accuracy=accuracy_score(y_test, predictions),
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        confusion=confusion_matrix(y_test, predictions),
        report=classification_report(
            y_test,
            predictions,
            target_names=[STAR_TYPE_NAMES[label] for label in sorted(y.unique())],
            zero_division=0,
        ),
    )

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

    return pipeline, report


def load_model(model_path: str | Path) -> Pipeline:
    """Load a previously saved pipeline from ``model_path``."""
    return joblib.load(model_path)


def predict_dataframe(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` augmented with ``predicted_type`` and ``predicted_label``."""
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing feature columns: {sorted(missing)}")
    predictions: Any = pipeline.predict(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    out = df.copy()
    out["predicted_type"] = predictions
    out["predicted_label"] = [STAR_TYPE_NAMES.get(int(p), str(p)) for p in predictions]
    return out
