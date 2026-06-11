"""Walk-forward ML validation for panel stock data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml.datasets import assert_no_label_leakage
from src.ml.metrics import classification_metrics
from src.ml.models import build_model_pipeline, predict_positive_probability


@dataclass(frozen=True)
class MLValidationResult:
    """Outputs from walk-forward validation."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame


def infer_horizon(label_column: str, default: int = 20) -> int:
    """Infer the forward horizon from label names such as label_outperform_20d."""

    for part in label_column.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return default


def date_walk_forward_splits(
    dates: pd.Series,
    train_window: int = 252,
    test_window: int = 63,
    step: int | None = None,
    embargo: int = 20,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Build date-based rolling train/test splits with an optional embargo gap."""

    unique_dates = pd.DatetimeIndex(pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values())
    if train_window <= 0 or test_window <= 0 or embargo < 0:
        raise ValueError("train_window and test_window must be positive; embargo must be non-negative")
    active_step = step or test_window
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    start = 0
    while start + train_window + embargo + test_window <= len(unique_dates):
        train_dates = unique_dates[start : start + train_window]
        test_start = start + train_window + embargo
        test_dates = unique_dates[test_start : test_start + test_window]
        splits.append((train_dates, test_dates))
        start += active_step
    return splits


def walk_forward_validate_classifier(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str = "logistic_regression",
    train_window: int = 252,
    test_window: int = 63,
    step: int | None = None,
    embargo: int = 20,
    probability_threshold: float = 0.5,
    random_state: int = 42,
) -> MLValidationResult:
    """Fit and score classifiers on strictly future walk-forward folds."""

    assert_no_label_leakage(feature_columns)
    if dataset.empty:
        return MLValidationResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    horizon = infer_horizon(label_column)

    splits = date_walk_forward_splits(
        dataset["Date"],
        train_window=train_window,
        test_window=test_window,
        step=step,
        embargo=embargo,
    )
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for fold, (train_dates, test_dates) in enumerate(splits, start=1):
        train = dataset[dataset["Date"].isin(train_dates)].dropna(subset=[label_column])
        test = dataset[dataset["Date"].isin(test_dates)].dropna(subset=[label_column])
        if train.empty or test.empty:
            continue

        y_train = train[label_column].astype(int)
        if y_train.nunique() < 2:
            probability = pd.Series(float(y_train.iloc[0]), index=test.index)
        else:
            model = build_model_pipeline(model_name, random_state=random_state)
            model.fit(train[feature_columns], y_train)
            probability = pd.Series(predict_positive_probability(model, test[feature_columns]), index=test.index)

        fold_predictions = pd.DataFrame(
            {
                "fold": fold,
                "Date": test["Date"],
                "Ticker": test["Ticker"],
                "actual": test[label_column].astype(int),
                "probability": probability,
                "prediction": (probability >= probability_threshold).astype(int),
                "forward_return": test.get(f"forward_{horizon}d_return"),
                "forward_excess_return": test.get(f"forward_{horizon}d_excess_return"),
                "forward_drawdown": test.get(f"forward_{horizon}d_drawdown"),
            },
            index=test.index,
        )
        prediction_frames.append(fold_predictions)
        row = classification_metrics(fold_predictions["actual"], fold_predictions["probability"], probability_threshold)
        row.update(
            {
                "fold": fold,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "train_rows": len(train),
                "test_rows": len(test),
                "positive_rate": float(fold_predictions["actual"].mean()),
            }
        )
        metric_rows.append(row)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    fold_metrics = pd.DataFrame(metric_rows)
    overall = (
        pd.DataFrame([classification_metrics(predictions["actual"], predictions["probability"], probability_threshold)])
        if not predictions.empty
        else pd.DataFrame()
    )
    return MLValidationResult(predictions=predictions, fold_metrics=fold_metrics, overall_metrics=overall)


def compare_feature_groups(
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    label_column: str,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    probability_threshold: float = 0.5,
) -> pd.DataFrame:
    """Run walk-forward validation for several feature groups."""

    rows: list[dict[str, object]] = []
    for group_name, columns in feature_groups.items():
        if not columns:
            continue
        result = walk_forward_validate_classifier(
            dataset,
            columns,
            label_column=label_column,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=probability_threshold,
        )
        if result.overall_metrics.empty:
            continue
        row = result.overall_metrics.iloc[0].to_dict()
        row.update({"feature_group": group_name, "features": len(columns), "folds": len(result.fold_metrics)})
        rows.append(row)
    return pd.DataFrame(rows)
