"""Walk-forward ML validation for panel stock data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml.datasets import assert_no_label_leakage
from src.ml.metrics import brier_score, classification_metrics
from src.ml.models import (
    available_model_candidates,
    build_classifier,
    build_model_pipeline,
    predict_positive_probability,
)


MODEL_SELECTION_MODES = ("current_default", "auto_select")


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


def prediction_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Return the safest common key set for validation prediction merges."""

    base_keys = ["Date", "Ticker"]
    if all(key in left and key in right for key in ["fold", *base_keys]):
        return ["fold", *base_keys]
    return base_keys


def deduplicate_prediction_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Deduplicate prediction rows before joining fold outputs."""

    if frame.empty or any(key not in frame for key in keys):
        return frame.copy()
    output = frame.copy()
    if "Date" in keys:
        output["Date"] = pd.to_datetime(output["Date"])
    return output.drop_duplicates(subset=keys, keep="last")


def _inner_train_validation_split(
    train: pd.DataFrame,
    *,
    validation_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an outer training period into past inner-train and later inner-validation rows."""

    unique_dates = pd.DatetimeIndex(pd.Series(pd.to_datetime(train["Date"])).drop_duplicates().sort_values())
    if len(unique_dates) < 4:
        return train.copy(), pd.DataFrame()
    validation_date_count = max(1, int(round(len(unique_dates) * validation_fraction)))
    validation_date_count = min(validation_date_count, len(unique_dates) - 1)
    validation_dates = unique_dates[-validation_date_count:]
    inner_train_dates = unique_dates[:-validation_date_count]
    inner_train = train[train["Date"].isin(inner_train_dates)].copy()
    inner_validation = train[train["Date"].isin(validation_dates)].copy()
    return inner_train, inner_validation


def _constant_probability(label: pd.Series, index: pd.Index) -> pd.Series:
    """Return a deterministic one-class probability fallback."""

    return pd.Series(float(label.iloc[0]) if len(label) else 0.0, index=index)


def _candidate_probability(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    candidate_name: str,
    random_state: int,
) -> pd.Series:
    """Fit one candidate on inner-train rows and score inner-validation rows."""

    y_train = train[label_column].astype(int)
    if y_train.nunique() < 2:
        return _constant_probability(y_train, validation.index)
    model = build_classifier(candidate_name, random_state=random_state)
    model.fit(train[feature_columns], y_train)
    return pd.Series(predict_positive_probability(model, validation[feature_columns]), index=validation.index)


def _selection_metric_name(metrics: dict[str, float]) -> str:
    for metric_name in ("roc_auc", "pr_auc", "brier_score", "accuracy"):
        value = metrics.get(metric_name)
        if pd.notna(value):
            return metric_name
    return "unavailable"


def _selection_sort_key(metrics: dict[str, float], candidate_order: int) -> tuple[float, float, float, float, int]:
    roc_auc = metrics.get("roc_auc")
    pr_auc = metrics.get("pr_auc")
    brier = metrics.get("brier_score")
    accuracy = metrics.get("accuracy")
    return (
        float(roc_auc) if pd.notna(roc_auc) else float("-inf"),
        float(pr_auc) if pd.notna(pr_auc) else float("-inf"),
        -float(brier) if pd.notna(brier) else float("-inf"),
        float(accuracy) if pd.notna(accuracy) else float("-inf"),
        -candidate_order,
    )


def _select_model_candidate(
    train: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    probability_threshold: float,
    random_state: int,
) -> dict[str, object]:
    """Select a candidate using only rows from the outer training period."""

    inner_train, inner_validation = _inner_train_validation_split(train)
    if inner_train.empty or inner_validation.empty:
        return {
            "selected_model": "current_default",
            "selection_metric": "insufficient_inner_validation",
            "inner_validation_rows": len(inner_validation),
            "inner_validation_positive_rate": pd.NA,
        }

    y_validation = inner_validation[label_column].astype(int)
    candidate_rows: list[dict[str, object]] = []
    for candidate_order, candidate_name in enumerate(available_model_candidates()):
        probability = _candidate_probability(
            inner_train,
            inner_validation,
            feature_columns,
            label_column,
            candidate_name,
            random_state,
        )
        metrics = classification_metrics(y_validation, probability, probability_threshold)
        metrics["brier_score"] = brier_score(y_validation, probability)
        candidate_rows.append(
            {
                "candidate": candidate_name,
                "candidate_order": candidate_order,
                "metrics": metrics,
                "sort_key": _selection_sort_key(metrics, candidate_order),
            }
        )

    best = max(candidate_rows, key=lambda row: row["sort_key"])
    selected_metrics = best["metrics"]
    return {
        "selected_model": best["candidate"],
        "selection_metric": _selection_metric_name(selected_metrics),
        "inner_validation_rows": len(inner_validation),
        "inner_validation_positive_rate": float(y_validation.mean()) if len(y_validation) else pd.NA,
    }


def summarize_model_selection(fold_metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    """Summarize selected walk-forward candidates for Research Lab display."""

    if fold_metrics.empty or "selected_model" not in fold_metrics:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for selected_model, group in fold_metrics.groupby("selected_model", dropna=False):
        rows.append(
            {
                "target": target,
                "folds": int(len(fold_metrics)),
                "selected_model": selected_model,
                "selection_count": int(len(group)),
                "mean_outer_roc_auc": group["roc_auc"].mean() if "roc_auc" in group else pd.NA,
                "mean_outer_pr_auc": group["pr_auc"].mean() if "pr_auc" in group else pd.NA,
                "mean_brier_score": group["brier_score"].mean() if "brier_score" in group else pd.NA,
                "interpretation": (
                    "Selected by inner training-period validation."
                    if "auto_select" in set(group.get("model_selection_mode", pd.Series(dtype=object)))
                    else "Fixed model used for all folds."
                ),
            }
        )
    return pd.DataFrame(rows)


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
    model_selection_mode: str = "current_default",
) -> MLValidationResult:
    """Fit and score classifiers on strictly future walk-forward folds."""

    assert_no_label_leakage(feature_columns)
    if model_selection_mode not in MODEL_SELECTION_MODES:
        raise ValueError(f"Unknown model_selection_mode: {model_selection_mode}")
    if dataset.empty:
        return MLValidationResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    horizon = infer_horizon(label_column)
    effective_embargo = max(int(embargo), int(horizon))

    splits = date_walk_forward_splits(
        dataset["Date"],
        train_window=train_window,
        test_window=test_window,
        step=step,
        embargo=effective_embargo,
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
            selection = {
                "selected_model": "constant_class",
                "selection_metric": "single_class_train",
                "inner_validation_rows": 0,
                "inner_validation_positive_rate": pd.NA,
            }
        else:
            selection = (
                _select_model_candidate(
                    train,
                    feature_columns,
                    label_column,
                    probability_threshold,
                    random_state,
                )
                if model_selection_mode == "auto_select"
                else {
                    "selected_model": model_name,
                    "selection_metric": "fixed_model",
                    "inner_validation_rows": 0,
                    "inner_validation_positive_rate": pd.NA,
                }
            )
            selected_model = str(selection["selected_model"])
            model = (
                build_classifier(selected_model, random_state=random_state)
                if model_selection_mode == "auto_select"
                else build_model_pipeline(model_name, random_state=random_state)
            )
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
                "selected_model": selection["selected_model"],
                "forward_return": test.get(f"forward_{horizon}d_return"),
                "forward_excess_return": test.get(f"forward_{horizon}d_excess_return"),
                "forward_drawdown": test.get(f"forward_{horizon}d_drawdown"),
                "forward_risk_adjusted_excess_return": test.get(
                    f"forward_{horizon}d_risk_adjusted_excess_return"
                ),
                "forward_tail_risk_adjusted_excess_return": test.get(
                    f"forward_{horizon}d_tail_risk_adjusted_excess_return"
                ),
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
                "requested_embargo": int(embargo),
                "effective_embargo": effective_embargo,
                "model_selection_mode": model_selection_mode,
                **selection,
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
    model_selection_mode: str = "current_default",
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
            model_selection_mode=model_selection_mode,
        )
        if result.overall_metrics.empty:
            continue
        row = result.overall_metrics.iloc[0].to_dict()
        row.update({"feature_group": group_name, "features": len(columns), "folds": len(result.fold_metrics)})
        rows.append(row)
    return pd.DataFrame(rows)
