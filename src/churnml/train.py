"""Train candidate models, select on validation data, report once on test."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .features import CATEGORICAL, NUMERIC, TARGET, make_preprocessor

FEATURE_COLUMNS = [*CATEGORICAL, *NUMERIC]

# MLflow saves scikit-learn models with skops, which refuses unknown types unless
# you list them. These two come from HistGradientBoosting and numpy; this code trains
# the model itself, so they are trusted. Never add types from files you did not make.
TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


@dataclass(frozen=True)
class TrainResult:
    best_model: str
    validation: dict[str, dict[str, float]]
    test: dict[str, float]
    run_id: str
    model_version: str | None = None


def candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("prep", make_preprocessor()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "hist_gb": Pipeline(
            [
                ("prep", make_preprocessor()),
                ("model", HistGradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }


def score(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics plus F1 at 0.5. Ranking quality matters most."""
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def split_data(
    features: pd.DataFrame, labels: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """60/20/20 stratified split. Returns X_train, X_val, X_test, y_*..."""
    data = features.merge(labels, on="customer_id", how="inner")
    x = data[FEATURE_COLUMNS]
    y = data[TARGET].astype(int)
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_and_log(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    tracking_uri: str,
    experiment: str = "churn",
    seed: int = 42,
    register_as: str | None = None,
) -> TrainResult:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(features, labels, seed)

    fitted: dict[str, Pipeline] = {}
    validation: dict[str, dict[str, float]] = {}
    with mlflow.start_run(run_name="selection") as parent:
        mlflow.log_params(
            {
                "seed": seed,
                "n_train": len(x_train),
                "n_val": len(x_val),
                "n_test": len(x_test),
                "features": ",".join(FEATURE_COLUMNS),
            }
        )
        for name, pipe in candidates(seed).items():
            with mlflow.start_run(run_name=name, nested=True):
                pipe.fit(x_train, y_train)
                metrics = score(y_val, pipe.predict_proba(x_val)[:, 1])
                mlflow.log_param("model", name)
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            fitted[name] = pipe
            validation[name] = metrics

        best = max(validation, key=lambda n: validation[n]["pr_auc"])
        test_metrics = score(y_test, fitted[best].predict_proba(x_test)[:, 1])
        mlflow.log_param("best_model", best)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        info = mlflow.sklearn.log_model(
            fitted[best],
            name="model",
            input_example=x_val.head(3),
            skops_trusted_types=TRUSTED_TYPES,
        )
        version = None
        if register_as:
            # Register the winner and point the "champion" alias at it, so a
            # serving app can load models:/<name>@champion without code changes.
            registered = mlflow.register_model(info.model_uri, register_as)
            version = str(registered.version)
            mlflow.MlflowClient().set_registered_model_alias(
                register_as, "champion", version
            )
        return TrainResult(best, validation, test_metrics, parent.info.run_id, version)
