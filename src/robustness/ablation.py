"""Feature-ablation comparison framework."""

from __future__ import annotations

import pandas as pd

from src.ml.datasets import feature_group_columns
from src.ml.metrics import score_quintile_analysis
from src.ml.validation import walk_forward_validate_classifier


def conclusion_from_metrics(metrics: pd.Series, quintiles: pd.DataFrame | None = None) -> str:
    """Classify whether a feature group appears useful from simple diagnostics."""

    roc_auc = metrics.get("roc_auc", pd.NA)
    pr_auc = metrics.get("pr_auc", pd.NA)
    f1 = metrics.get("f1", 0.0)
    quintile_spread = pd.NA
    if quintiles is not None and not quintiles.empty and "average_forward_excess_return" in quintiles:
        ordered = quintiles.sort_values("score_quintile")
        quintile_spread = ordered["average_forward_excess_return"].iloc[-1] - ordered["average_forward_excess_return"].iloc[0]

    if pd.notna(roc_auc) and roc_auc >= 0.58 and f1 > 0.45 and (pd.isna(quintile_spread) or quintile_spread > 0):
        return "adds value"
    if pd.notna(roc_auc) and roc_auc < 0.50 and pd.notna(quintile_spread) and quintile_spread < 0:
        return "likely overfit"
    if pd.notna(roc_auc) and roc_auc < 0.52 and f1 < 0.35:
        return "no clear value"
    return "mixed"


def run_feature_ablation(
    dataset: pd.DataFrame,
    label_column: str,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    probability_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compare technical, Fourier, wavelet, combined, and rule-baseline variants."""

    rows: list[dict[str, object]] = []
    for group in ("technical", "technical_fourier", "technical_wavelet", "all"):
        columns = feature_group_columns(dataset, group)
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
        quintiles = score_quintile_analysis(result.predictions)
        row = result.overall_metrics.iloc[0].to_dict()
        row.update(
            {
                "system": group,
                "features": len(columns),
                "folds": len(result.fold_metrics),
                "quintile_spread": (
                    quintiles["average_forward_excess_return"].iloc[-1]
                    - quintiles["average_forward_excess_return"].iloc[0]
                    if not quintiles.empty
                    else pd.NA
                ),
                "conclusion": conclusion_from_metrics(result.overall_metrics.iloc[0], quintiles),
            }
        )
        rows.append(row)

    if "regime" in dataset and label_column in dataset:
        rule_data = dataset.dropna(subset=[label_column]).copy()
        if not rule_data.empty:
            rule_data["probability"] = rule_data["regime"].astype(str).str.startswith("Uptrend").astype(float)
            accuracy = (rule_data["probability"].astype(int) == rule_data[label_column].astype(int)).mean()
            rows.append(
                {
                    "system": "rule_based_regime_only",
                    "features": 0,
                    "folds": 0,
                    "accuracy": accuracy,
                    "precision": pd.NA,
                    "recall": pd.NA,
                    "f1": pd.NA,
                    "roc_auc": pd.NA,
                    "pr_auc": pd.NA,
                    "quintile_spread": pd.NA,
                    "conclusion": "baseline",
                }
            )
    return pd.DataFrame(rows)

