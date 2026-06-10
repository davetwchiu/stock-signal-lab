"""Validation metrics and score-bucket analysis for Stage 2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    """Compute robust binary classification metrics."""

    truth = y_true.dropna().astype(int)
    prob = y_prob.reindex(truth.index).astype(float)
    pred = (prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(truth, pred),
        "precision": precision_score(truth, pred, zero_division=0),
        "recall": recall_score(truth, pred, zero_division=0),
        "f1": f1_score(truth, pred, zero_division=0),
    }
    if truth.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(truth, prob)
        metrics["pr_auc"] = average_precision_score(truth, prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def confusion_matrix_frame(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> pd.DataFrame:
    """Return a labeled 2x2 confusion matrix."""

    truth = y_true.dropna().astype(int)
    pred = (y_prob.reindex(truth.index) >= threshold).astype(int)
    matrix = confusion_matrix(truth, pred, labels=[0, 1])
    return pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])


def calibration_table(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    label_column: str = "actual",
    bins: int = 10,
) -> pd.DataFrame:
    """Summarize observed hit rate by probability bucket."""

    data = predictions.dropna(subset=[probability_column, label_column]).copy()
    if data.empty:
        return pd.DataFrame()
    data["probability_bin"] = pd.cut(data[probability_column], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return (
        data.groupby("probability_bin", observed=True)
        .agg(
            count=(label_column, "size"),
            average_probability=(probability_column, "mean"),
            observed_rate=(label_column, "mean"),
        )
        .reset_index()
    )


def score_quintile_analysis(
    predictions: pd.DataFrame,
    score_column: str = "probability",
    label_column: str = "actual",
    forward_return_column: str = "forward_return",
    forward_excess_column: str = "forward_excess_return",
    forward_drawdown_column: str = "forward_drawdown",
) -> pd.DataFrame:
    """Compare forward outcomes across model-score quintiles."""

    required = [score_column, label_column, forward_return_column, forward_excess_column, forward_drawdown_column]
    data = predictions.dropna(subset=[column for column in required if column in predictions]).copy()
    if data.empty or data[score_column].nunique() < 2:
        return pd.DataFrame()
    data["score_quintile"] = pd.qcut(data[score_column], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    return (
        data.groupby("score_quintile", observed=True)
        .agg(
            count=(label_column, "size"),
            average_score=(score_column, "mean"),
            hit_rate=(label_column, "mean"),
            average_forward_return=(forward_return_column, "mean"),
            average_forward_excess_return=(forward_excess_column, "mean"),
            average_forward_drawdown=(forward_drawdown_column, "mean"),
        )
        .reset_index()
    )

