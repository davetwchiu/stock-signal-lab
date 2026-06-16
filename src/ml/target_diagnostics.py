"""Diagnostics-only helpers for comparing alternative ML target definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.ml.labels import make_forward_labels
from src.ml.metrics import calibration_summary
from src.ml.validation import walk_forward_validate_classifier


MIN_TARGET_DIAGNOSTIC_SAMPLE_COUNT = 50
MIN_TARGET_BUCKET_COUNT = 5
RECENT_VOLATILITY_FLOOR = 1.0e-6

TARGET_DEFINITION_COLUMNS = [
    "target_id",
    "display_name",
    "label_column",
    "target_type",
    "horizon",
    "description",
    "positive_label_meaning",
]

TARGET_BALANCE_COLUMNS = [
    "target_id",
    "display_name",
    "horizon",
    "sample_count",
    "positive_rate",
    "class_balance_status",
    "mean_forward_excess_return_when_positive",
    "mean_forward_excess_return_when_negative",
    "mean_forward_drawdown_when_positive",
    "mean_forward_drawdown_when_negative",
    "too_few_samples",
    "too_rare_positive_class",
    "too_common_positive_class",
]

TARGET_WALK_FORWARD_COLUMNS = [
    "target_id",
    "display_name",
    "folds",
    "prediction_count",
    "positive_rate",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "calibration_gap",
    "top_bucket_positive_rate",
    "bottom_bucket_positive_rate",
    "bucket_spread",
    "quality_summary",
    "interpretation",
]


@dataclass(frozen=True)
class TargetCandidate:
    """One diagnostics-only target candidate."""

    target_id: str
    display_name: str
    label_column: str
    target_type: str
    horizon: int
    description: str
    positive_label_meaning: str


def target_candidate_registry(base_horizon: int = 20) -> list[TargetCandidate]:
    """Return explicit diagnostics-only target candidates for Research Lab."""

    return [
        TargetCandidate(
            target_id="outperform_20d",
            display_name="Current 20d outperformance",
            label_column=f"label_outperform_{base_horizon}d",
            target_type="production_baseline",
            horizon=base_horizon,
            description="Current production target baseline.",
            positive_label_meaning=(
                f"Future {base_horizon}d excess return over the benchmark exceeds the current threshold."
            ),
        ),
        TargetCandidate(
            target_id="outperform_60d",
            display_name="Longer 60d outperformance",
            label_column="label_outperform_60d",
            target_type="longer_horizon_outperformance",
            horizon=60,
            description="Longer-horizon benchmark-relative target.",
            positive_label_meaning="Future 60d excess return over the benchmark exceeds the current threshold.",
        ),
        TargetCandidate(
            target_id="risk_adjusted_excess_20d",
            display_name="Recent-vol adjusted excess",
            label_column="label_risk_adjusted_excess_20d",
            target_type="risk_adjusted_excess",
            horizon=base_horizon,
            description="Future excess return divided by recent realized volatility.",
            positive_label_meaning=(
                f"Future {base_horizon}d excess return is positive after scaling by recent realized volatility."
            ),
        ),
        TargetCandidate(
            target_id="top_tercile_excess_20d",
            display_name="Top-third relative performer",
            label_column="label_top_tercile_excess_20d",
            target_type="cross_sectional_relative",
            horizon=base_horizon,
            description="Same-date cross-sectional top-third future excess return target.",
            positive_label_meaning=(
                f"Ticker is in the top third of the same-date universe by future {base_horizon}d excess return."
            ),
        ),
        TargetCandidate(
            target_id="tail_adjusted_outperform_20d",
            display_name="Tail-adjusted outperformance",
            label_column="label_tail_adjusted_outperform_20d",
            target_type="tail_adjusted_outperformance",
            horizon=base_horizon,
            description="Future excess return penalized by forward drawdown.",
            positive_label_meaning=(
                f"Future {base_horizon}d excess return plus forward drawdown exceeds the current threshold."
            ),
        ),
        TargetCandidate(
            target_id="pullback_recovery_20d",
            display_name="Pullback and recovery",
            label_column="label_pullback_recovery_20d",
            target_type="pullback_recovery",
            horizon=base_horizon,
            description="Forward pullback that still recovers to non-negative benchmark-relative return.",
            positive_label_meaning=(
                f"Ticker has a material forward drawdown but finishes with non-negative {base_horizon}d excess return."
            ),
        ),
    ]


def target_definition_table(candidates: list[TargetCandidate] | None = None) -> pd.DataFrame:
    """Return a display table for diagnostics-only target definitions."""

    active = candidates or target_candidate_registry()
    return pd.DataFrame([asdict(candidate) for candidate in active], columns=TARGET_DEFINITION_COLUMNS)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _recent_realized_volatility(group: pd.DataFrame, horizon: int) -> pd.Series:
    if f"volatility_{horizon}d" in group:
        volatility = _numeric(group[f"volatility_{horizon}d"])
    elif "daily_return" in group:
        volatility = _numeric(group["daily_return"]).rolling(horizon, min_periods=max(2, horizon // 2)).std()
    else:
        volatility = pd.Series(np.nan, index=group.index)
    return volatility.where(volatility.abs() > RECENT_VOLATILITY_FLOOR)


def _ensure_horizon_labels(
    output: pd.DataFrame,
    benchmark_price: pd.Series,
    *,
    horizon: int,
    outperformance_threshold: float,
    drawdown_threshold: float,
) -> pd.DataFrame:
    required = {
        f"forward_{horizon}d_excess_return",
        f"forward_{horizon}d_drawdown",
        f"label_outperform_{horizon}d",
    }
    if required.issubset(output.columns):
        return output

    for _, group in output.groupby("Ticker", sort=False):
        ordered = group.sort_values("Date")
        features = ordered.set_index(pd.to_datetime(ordered["Date"]))
        labels = make_forward_labels(
            features,
            benchmark_price=benchmark_price,
            horizon=horizon,
            outperformance_threshold=outperformance_threshold,
            drawdown_threshold=drawdown_threshold,
        )
        for column in labels.columns:
            if column not in output:
                output[column] = pd.NA
            output.loc[ordered.index, column] = labels[column].to_numpy()
    return output


def add_target_candidate_labels(
    panel: pd.DataFrame,
    benchmark_price: pd.Series,
    *,
    base_horizon: int = 20,
    outperformance_threshold: float = 0.02,
    drawdown_threshold: float = -0.10,
    pullback_drawdown_threshold: float = -0.05,
    min_cross_sectional_count: int = 3,
) -> pd.DataFrame:
    """Add diagnostics-only target columns without changing production labels."""

    if panel.empty or "Ticker" not in panel or "Date" not in panel or "Adj Close" not in panel:
        return panel.copy()

    output = panel.copy()
    output["Date"] = pd.to_datetime(output["Date"])
    benchmark = benchmark_price.copy()
    benchmark.index = pd.to_datetime(benchmark.index)
    output = _ensure_horizon_labels(
        output,
        benchmark,
        horizon=base_horizon,
        outperformance_threshold=outperformance_threshold,
        drawdown_threshold=drawdown_threshold,
    )
    output = _ensure_horizon_labels(
        output,
        benchmark,
        horizon=60,
        outperformance_threshold=outperformance_threshold,
        drawdown_threshold=drawdown_threshold,
    )

    risk_adjusted_col = f"forward_{base_horizon}d_recent_vol_adjusted_excess_return"
    tail_adjusted_col = f"forward_{base_horizon}d_tail_adjusted_score"
    excess_col = f"forward_{base_horizon}d_excess_return"
    drawdown_col = f"forward_{base_horizon}d_drawdown"
    output[risk_adjusted_col] = pd.NA
    output[tail_adjusted_col] = _numeric(output[excess_col]) + _numeric(output[drawdown_col])

    for _, group in output.groupby("Ticker", sort=False):
        ordered = group.sort_values("Date")
        volatility = _recent_realized_volatility(ordered, base_horizon)
        risk_adjusted = _numeric(ordered[excess_col]) / volatility
        output.loc[ordered.index, risk_adjusted_col] = risk_adjusted.to_numpy()

    risk_adjusted = _numeric(output[risk_adjusted_col])
    output["label_risk_adjusted_excess_20d"] = (risk_adjusted > 0.0).astype(float)
    output.loc[risk_adjusted.isna(), "label_risk_adjusted_excess_20d"] = np.nan

    tail_adjusted = _numeric(output[tail_adjusted_col])
    output["label_tail_adjusted_outperform_20d"] = (tail_adjusted > outperformance_threshold).astype(float)
    output.loc[tail_adjusted.isna(), "label_tail_adjusted_outperform_20d"] = np.nan

    drawdown = _numeric(output[drawdown_col])
    excess = _numeric(output[excess_col])
    pullback = (drawdown <= pullback_drawdown_threshold) & (excess >= 0.0)
    output["label_pullback_recovery_20d"] = pullback.astype(float)
    output.loc[drawdown.isna() | excess.isna(), "label_pullback_recovery_20d"] = np.nan

    output["label_top_tercile_excess_20d"] = np.nan
    valid_excess = output.dropna(subset=[excess_col])
    for _, date_group in valid_excess.groupby("Date", sort=False):
        if len(date_group) < min_cross_sectional_count:
            continue
        ranks = _numeric(date_group[excess_col]).rank(method="first", pct=True)
        output.loc[date_group.index, "label_top_tercile_excess_20d"] = (ranks > (2.0 / 3.0)).astype(float)

    return output


def _class_balance_status(positive_rate: object) -> str:
    if pd.isna(positive_rate):
        return "Unusable"
    rate = float(positive_rate)
    if rate < 0.05 or rate > 0.95:
        return "Unusable"
    if rate < 0.20 or rate > 0.80:
        return "Skewed"
    return "Healthy"


def _candidate_forward_columns(candidate: TargetCandidate) -> tuple[str, str]:
    return f"forward_{candidate.horizon}d_excess_return", f"forward_{candidate.horizon}d_drawdown"


def build_target_balance_diagnostics(
    panel: pd.DataFrame,
    candidates: list[TargetCandidate] | None = None,
    *,
    min_sample_count: int = MIN_TARGET_DIAGNOSTIC_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Summarize target balance and realized outcomes for candidate labels."""

    active = candidates or target_candidate_registry()
    rows: list[dict[str, object]] = []
    for candidate in active:
        label_col = candidate.label_column
        if panel.empty or label_col not in panel:
            sample_count = 0
            positive_rate = pd.NA
            positives = pd.Series(dtype=bool)
            data = pd.DataFrame()
        else:
            label = _numeric(panel[label_col])
            data = panel.assign(_candidate_label=label).dropna(subset=["_candidate_label"]).copy()
            sample_count = int(len(data))
            positive_rate = float(data["_candidate_label"].mean()) if sample_count else pd.NA
            positives = data["_candidate_label"] == 1.0

        excess_col, drawdown_col = _candidate_forward_columns(candidate)
        status = _class_balance_status(positive_rate)
        rows.append(
            {
                "target_id": candidate.target_id,
                "display_name": candidate.display_name,
                "horizon": candidate.horizon,
                "sample_count": sample_count,
                "positive_rate": positive_rate,
                "class_balance_status": status,
                "mean_forward_excess_return_when_positive": (
                    _numeric(data.loc[positives, excess_col]).mean()
                    if sample_count and excess_col in data
                    else pd.NA
                ),
                "mean_forward_excess_return_when_negative": (
                    _numeric(data.loc[~positives, excess_col]).mean()
                    if sample_count and excess_col in data
                    else pd.NA
                ),
                "mean_forward_drawdown_when_positive": (
                    _numeric(data.loc[positives, drawdown_col]).mean()
                    if sample_count and drawdown_col in data
                    else pd.NA
                ),
                "mean_forward_drawdown_when_negative": (
                    _numeric(data.loc[~positives, drawdown_col]).mean()
                    if sample_count and drawdown_col in data
                    else pd.NA
                ),
                "too_few_samples": sample_count < min_sample_count,
                "too_rare_positive_class": bool(pd.notna(positive_rate) and float(positive_rate) < 0.05),
                "too_common_positive_class": bool(pd.notna(positive_rate) and float(positive_rate) > 0.95),
            }
        )
    return pd.DataFrame(rows, columns=TARGET_BALANCE_COLUMNS)


def _bucket_positive_rates(predictions: pd.DataFrame, min_bucket_count: int) -> tuple[object, object, object]:
    if predictions.empty or "probability" not in predictions or "actual" not in predictions:
        return pd.NA, pd.NA, pd.NA
    data = predictions[["probability", "actual"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < min_bucket_count * 2 or data["probability"].nunique(dropna=True) < 2:
        return pd.NA, pd.NA, pd.NA
    try:
        data["_bucket"] = pd.qcut(data["probability"], q=3, labels=False, duplicates="drop")
    except ValueError:
        return pd.NA, pd.NA, pd.NA
    data = data.dropna(subset=["_bucket"])
    if data["_bucket"].nunique(dropna=True) < 2:
        return pd.NA, pd.NA, pd.NA
    grouped = data.groupby("_bucket", observed=True)["actual"]
    counts = grouped.size()
    rates = grouped.mean().sort_index()
    low = rates.index.min()
    high = rates.index.max()
    if counts.loc[low] < min_bucket_count or counts.loc[high] < min_bucket_count:
        return pd.NA, pd.NA, pd.NA
    bottom_rate = float(rates.loc[low])
    top_rate = float(rates.loc[high])
    return top_rate, bottom_rate, top_rate - bottom_rate


def _target_quality_summary(balance_row: pd.Series, wf_row: dict[str, object]) -> tuple[str, str]:
    if (
        bool(balance_row.get("too_few_samples"))
        or bool(balance_row.get("too_rare_positive_class"))
        or bool(balance_row.get("too_common_positive_class"))
        or int(wf_row.get("prediction_count") or 0) == 0
    ):
        return "Unusable", "Unusable diagnostic target in this sample; do not consider it for production."

    positive_rate = float(balance_row.get("positive_rate"))
    roc_auc = pd.to_numeric(pd.Series([wf_row.get("roc_auc")]), errors="coerce").iloc[0]
    pr_auc = pd.to_numeric(pd.Series([wf_row.get("pr_auc")]), errors="coerce").iloc[0]
    bucket_spread = pd.to_numeric(pd.Series([wf_row.get("bucket_spread")]), errors="coerce").iloc[0]
    pr_lift = pr_auc - positive_rate if pd.notna(pr_auc) else pd.NA

    if (
        (pd.notna(roc_auc) and roc_auc >= 0.55)
        or (pd.notna(pr_lift) and pr_lift >= 0.05)
        or (pd.notna(bucket_spread) and bucket_spread >= 0.10)
    ):
        return "Promising", "Promising diagnostic target, not yet production target."
    if (
        (pd.notna(roc_auc) and roc_auc < 0.52)
        and (pd.isna(pr_lift) or pr_lift < 0.03)
        and (pd.isna(bucket_spread) or bucket_spread < 0.05)
    ):
        return "Weak", "Weak separation in this walk-forward sample."
    return "Mixed", "Mixed diagnostic evidence; useful for review but not decisive."


def build_target_walk_forward_comparison(
    panel: pd.DataFrame,
    feature_columns: list[str],
    candidates: list[TargetCandidate] | None = None,
    *,
    model_name: str = "logistic_regression",
    train_window: int = 252,
    test_window: int = 63,
    step: int | None = None,
    embargo: int = 20,
    probability_threshold: float = 0.5,
    model_selection_mode: str = "current_default",
    min_sample_count: int = MIN_TARGET_DIAGNOSTIC_SAMPLE_COUNT,
    min_bucket_count: int = MIN_TARGET_BUCKET_COUNT,
) -> pd.DataFrame:
    """Run light walk-forward diagnostics for each candidate target."""

    active = candidates or target_candidate_registry()
    balance = build_target_balance_diagnostics(panel, active, min_sample_count=min_sample_count).set_index("target_id")
    rows: list[dict[str, object]] = []
    for candidate in active:
        balance_row = balance.loc[candidate.target_id]
        row: dict[str, object] = {
            "target_id": candidate.target_id,
            "display_name": candidate.display_name,
            "folds": 0,
            "prediction_count": 0,
            "positive_rate": balance_row["positive_rate"],
            "roc_auc": pd.NA,
            "pr_auc": pd.NA,
            "brier_score": pd.NA,
            "calibration_gap": pd.NA,
            "top_bucket_positive_rate": pd.NA,
            "bottom_bucket_positive_rate": pd.NA,
            "bucket_spread": pd.NA,
        }
        if (
            candidate.label_column in panel
            and feature_columns
            and not bool(balance_row["too_few_samples"])
            and not bool(balance_row["too_rare_positive_class"])
            and not bool(balance_row["too_common_positive_class"])
        ):
            result = walk_forward_validate_classifier(
                panel,
                feature_columns,
                label_column=candidate.label_column,
                model_name=model_name,
                train_window=train_window,
                test_window=test_window,
                step=step,
                embargo=embargo,
                probability_threshold=probability_threshold,
                model_selection_mode=model_selection_mode,
            )
            row["folds"] = int(len(result.fold_metrics))
            row["prediction_count"] = int(len(result.predictions))
            if not result.predictions.empty:
                row["positive_rate"] = float(result.predictions["actual"].mean())
                if not result.overall_metrics.empty:
                    for metric in ("roc_auc", "pr_auc", "brier_score"):
                        row[metric] = result.overall_metrics.iloc[0].get(metric, pd.NA)
                calibration = calibration_summary(result.predictions)
                if not calibration.empty:
                    row["calibration_gap"] = calibration.iloc[0].get("calibration_gap", pd.NA)
                top_rate, bottom_rate, spread = _bucket_positive_rates(result.predictions, min_bucket_count)
                row["top_bucket_positive_rate"] = top_rate
                row["bottom_bucket_positive_rate"] = bottom_rate
                row["bucket_spread"] = spread

        summary, interpretation = _target_quality_summary(balance_row, row)
        row["quality_summary"] = summary
        row["interpretation"] = interpretation
        rows.append(row)
    return pd.DataFrame(rows, columns=TARGET_WALK_FORWARD_COLUMNS)
