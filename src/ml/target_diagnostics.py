"""Diagnostics-only helpers for comparing alternative ML target definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.ml.labels import make_forward_labels
from src.ml.metrics import calibration_summary, classification_metrics
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

TARGET_FEATURE_GROUP_COMPARISON_COLUMNS = [
    "target_id",
    "display_name",
    "feature_group",
    "folds",
    "prediction_count",
    "positive_rate",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "calibration_gap",
    "bucket_spread",
    "quality_summary",
    "interpretation",
]

TARGET_REGIME_COMPARISON_COLUMNS = [
    "target_id",
    "display_name",
    "regime",
    "sample_size",
    "positive_rate",
    "roc_auc",
    "pr_auc",
    "top_bucket_positive_rate",
    "bottom_bucket_positive_rate",
    "bucket_spread",
    "direction",
    "quality_summary",
    "interpretation",
]

TARGET_STABILITY_SUMMARY_COLUMNS = [
    "target_id",
    "display_name",
    "best_feature_group",
    "best_feature_group_metric",
    "feature_group_consistency",
    "regime_positive_count",
    "regime_negative_count",
    "worst_regime_bucket_spread",
    "overall_stability",
    "next_step_candidate",
    "interpretation",
]

TARGET_QUALITY_SUMMARY_COLUMNS = [
    "target_id",
    "candidate_rank",
    "display_name",
    "sample_count",
    "positive_rate",
    "class_balance_status",
    "overall_auc",
    "overall_pr_auc",
    "overall_brier_score",
    "overall_calibration_gap",
    "overall_bucket_spread",
    "best_feature_group",
    "best_feature_group_auc",
    "feature_group_consistency",
    "regime_stability",
    "regime_inversion_count",
    "worst_regime_bucket_spread",
    "calibration_quality",
    "bucket_separation_quality",
    "overall_target_quality",
    "production_candidate_status",
    "recommended_next_step",
    "interpretation",
]

TARGET_ARENA_TARGET_IDS = (
    "outperform_20d",
    "top_tercile_excess_20d",
    "risk_adjusted_excess_20d",
    "drawdown_adjusted_opportunity_20d",
    "pullback_recovery_20d",
)

TARGET_ARENA_COLUMNS = [
    "target_id",
    "display_name",
    "sample_count",
    "label_prevalence",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "bucket_spread",
    "calibration_gap",
    "regime_inversion_count",
    "feature_group_stability",
    "evidence_classification",
    "arena_decision",
    "rejection_reason",
    "future_production_experiment",
    "suggested_next_research",
]

DEFAULT_TARGET_FEATURE_GROUPS = ("technical", "technical_fourier", "technical_wavelet", "all")
DEFAULT_TARGET_REGIME_COLUMNS = ("regime",)


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
    output["label_drawdown_adjusted_opportunity_20d"] = output["label_tail_adjusted_outperform_20d"]

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


def _target_walk_forward_row_and_predictions(
    panel: pd.DataFrame,
    feature_columns: list[str],
    candidate: TargetCandidate,
    balance_row: pd.Series,
    *,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    probability_threshold: float,
    model_selection_mode: str,
    min_bucket_count: int,
) -> tuple[dict[str, object], pd.DataFrame]:
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
    predictions = pd.DataFrame()
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
        predictions = result.predictions
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
    return row, predictions


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
        row, _ = _target_walk_forward_row_and_predictions(
            panel,
            feature_columns,
            candidate,
            balance_row,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=probability_threshold,
            model_selection_mode=model_selection_mode,
            min_bucket_count=min_bucket_count,
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=TARGET_WALK_FORWARD_COLUMNS)


def build_target_feature_group_comparison(
    panel: pd.DataFrame,
    feature_groups: dict[str, list[str]],
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
    """Compare diagnostics-only target candidates across existing feature groups."""

    active = candidates or target_candidate_registry()
    rows: list[dict[str, object]] = []
    for feature_group, columns in feature_groups.items():
        if not columns:
            continue
        comparison = build_target_walk_forward_comparison(
            panel,
            columns,
            active,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=probability_threshold,
            model_selection_mode=model_selection_mode,
            min_sample_count=min_sample_count,
            min_bucket_count=min_bucket_count,
        )
        for row in comparison.to_dict("records"):
            row["feature_group"] = feature_group
            rows.append(row)
    return pd.DataFrame(rows, columns=TARGET_FEATURE_GROUP_COMPARISON_COLUMNS)


def _merge_prediction_regimes(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    regime_columns: tuple[str, ...],
) -> pd.DataFrame:
    if predictions.empty or panel.empty or "Date" not in predictions or "Ticker" not in predictions:
        return pd.DataFrame()
    available = [column for column in regime_columns if column in panel]
    if not available:
        return pd.DataFrame()
    lookup = panel[["Date", "Ticker", *available]].copy()
    lookup["Date"] = pd.to_datetime(lookup["Date"])
    lookup = lookup.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    output = predictions.copy()
    output["Date"] = pd.to_datetime(output["Date"])
    return output.merge(lookup, on=["Date", "Ticker"], how="left")


def _target_regime_direction(bucket_spread: object, sample_size: int, min_sample_count: int) -> tuple[str, str, str]:
    spread = pd.to_numeric(pd.Series([bucket_spread]), errors="coerce").iloc[0]
    if sample_size < min_sample_count or pd.isna(spread):
        return (
            "insufficient",
            "Insufficient",
            "Regime sample is too small or bucket separation is unavailable for this target.",
        )
    if float(spread) >= 0.05:
        return (
            "positive",
            "Positive separation",
            "Target probabilities separate higher-positive-rate buckets in this regime.",
        )
    if float(spread) <= -0.05:
        return (
            "inverted",
            "Inverted",
            "Target probabilities are inverted in this regime.",
        )
    return (
        "flat",
        "Weak",
        "Target probabilities show weak or flat separation in this regime.",
    )


def _target_regime_rows(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    candidate: TargetCandidate,
    *,
    regime_columns: tuple[str, ...],
    min_sample_count: int,
    min_bucket_count: int,
) -> list[dict[str, object]]:
    data = _merge_prediction_regimes(predictions, panel, regime_columns)
    if data.empty:
        return []
    rows: list[dict[str, object]] = []
    available = [column for column in regime_columns if column in data]
    for regime_col in available:
        regime_data = data.dropna(subset=[regime_col]).copy()
        if regime_data.empty:
            continue
        regime_data[regime_col] = regime_data[regime_col].astype(str).str.strip()
        regime_data = regime_data[regime_data[regime_col] != ""]
        for regime, group in regime_data.groupby(regime_col, sort=True):
            metrics = classification_metrics(group["actual"], group["probability"])
            top_rate, bottom_rate, spread = _bucket_positive_rates(group, min_bucket_count)
            direction, summary, interpretation = _target_regime_direction(
                spread,
                int(len(group)),
                min_sample_count,
            )
            if group["actual"].nunique(dropna=True) < 2:
                interpretation += " ROC-AUC is unavailable because this regime has one class."
            rows.append(
                {
                    "target_id": candidate.target_id,
                    "display_name": candidate.display_name,
                    "regime": regime,
                    "sample_size": int(len(group)),
                    "positive_rate": float(group["actual"].mean()) if len(group) else pd.NA,
                    "roc_auc": metrics.get("roc_auc", pd.NA),
                    "pr_auc": metrics.get("pr_auc", pd.NA),
                    "top_bucket_positive_rate": top_rate,
                    "bottom_bucket_positive_rate": bottom_rate,
                    "bucket_spread": spread,
                    "direction": direction,
                    "quality_summary": summary,
                    "interpretation": interpretation,
                }
            )
    return rows


def build_target_regime_comparison(
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
    regime_columns: tuple[str, ...] = DEFAULT_TARGET_REGIME_COLUMNS,
    min_sample_count: int = MIN_TARGET_BUCKET_COUNT,
    min_target_sample_count: int = MIN_TARGET_DIAGNOSTIC_SAMPLE_COUNT,
    min_bucket_count: int = MIN_TARGET_BUCKET_COUNT,
) -> pd.DataFrame:
    """Segment out-of-sample target predictions by existing regime labels."""

    active = candidates or target_candidate_registry()
    balance = build_target_balance_diagnostics(panel, active, min_sample_count=min_target_sample_count).set_index(
        "target_id"
    )
    rows: list[dict[str, object]] = []
    for candidate in active:
        _, predictions = _target_walk_forward_row_and_predictions(
            panel,
            feature_columns,
            candidate,
            balance.loc[candidate.target_id],
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=probability_threshold,
            model_selection_mode=model_selection_mode,
            min_bucket_count=min_bucket_count,
        )
        rows.extend(
            _target_regime_rows(
                predictions,
                panel,
                candidate,
                regime_columns=regime_columns,
                min_sample_count=min_sample_count,
                min_bucket_count=min_bucket_count,
            )
        )
    return pd.DataFrame(rows, columns=TARGET_REGIME_COMPARISON_COLUMNS)


def _numeric_value(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _feature_group_score(row: pd.Series) -> float:
    spread = _numeric_value(row.get("bucket_spread"))
    roc_auc = _numeric_value(row.get("roc_auc"))
    pr_auc = _numeric_value(row.get("pr_auc"))
    positive_rate = _numeric_value(row.get("positive_rate"))
    scores = []
    if not np.isnan(spread):
        scores.append(spread)
    if not np.isnan(roc_auc):
        scores.append((roc_auc - 0.5) * 2.0)
    if not np.isnan(pr_auc) and not np.isnan(positive_rate):
        scores.append(pr_auc - positive_rate)
    return max(scores) if scores else float("-inf")


def _feature_group_direction(row: pd.Series) -> str:
    prediction_count = pd.to_numeric(pd.Series([row.get("prediction_count")]), errors="coerce").fillna(0).iloc[0]
    if int(prediction_count) <= 0 or row.get("quality_summary") == "Unusable":
        return "insufficient"
    score = _feature_group_score(row)
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "inverted"
    return "flat"


def _feature_group_consistency(group: pd.DataFrame) -> str:
    directions = [_feature_group_direction(row) for _, row in group.iterrows()]
    supported = [direction for direction in directions if direction != "insufficient"]
    if not supported:
        return "No usable feature groups"
    positives = supported.count("positive")
    inverted = supported.count("inverted")
    flat = supported.count("flat")
    if positives == len(supported):
        return "Consistent positive"
    if positives and inverted == 0 and flat:
        return "Mostly positive"
    if positives and (inverted or flat):
        return "Feature-group dependent"
    if inverted:
        return "Inverted or unstable"
    return "Weak"


def _stability_label(
    *,
    best_score: float,
    feature_group_consistency: str,
    regime_positive_count: int,
    regime_negative_count: int,
) -> tuple[str, str, str]:
    if best_score == float("-inf"):
        return "Unusable", "No", "Unusable target in this sample."
    if regime_negative_count > 0 and best_score >= 0.05:
        return (
            "Promising but regime-sensitive",
            "Review",
            "Promising but regime-sensitive: this target shows useful separation overall, but weak or inverted results in some regimes.",
        )
    if feature_group_consistency in {"Feature-group dependent", "Inverted or unstable"} and best_score >= 0.05:
        return (
            "Feature-group dependent",
            "Review",
            "Feature-group dependent: this target works better with some feature sets than others.",
        )
    if best_score >= 0.10 and feature_group_consistency in {"Consistent positive", "Mostly positive"}:
        return (
            "Strong candidate",
            "Yes",
            "Strong candidate: this target has useful separation across feature sets without obvious regime inversion.",
        )
    if best_score >= 0.05 or regime_positive_count > 0:
        return (
            "Promising but regime-sensitive",
            "Review",
            "Promising but needs more evidence across feature groups and regimes.",
        )
    return "Weak", "No", "Weak target evidence in this sample."


def build_target_stability_summary(
    feature_group_comparison: pd.DataFrame,
    regime_comparison: pd.DataFrame,
    candidates: list[TargetCandidate] | None = None,
) -> pd.DataFrame:
    """Summarize target stability across feature groups and regimes."""

    active = candidates or target_candidate_registry()
    rows: list[dict[str, object]] = []
    feature_group_order = {name: index for index, name in enumerate(DEFAULT_TARGET_FEATURE_GROUPS)}
    for candidate in active:
        fg_rows = feature_group_comparison[
            feature_group_comparison.get("target_id", pd.Series(dtype=object)) == candidate.target_id
        ].copy()
        if fg_rows.empty:
            best_feature_group = pd.NA
            best_metric = pd.NA
            best_score = float("-inf")
            consistency = "No usable feature groups"
        else:
            fg_rows["_score"] = fg_rows.apply(_feature_group_score, axis=1)
            fg_rows["_order"] = fg_rows["feature_group"].map(feature_group_order).fillna(len(feature_group_order))
            best = fg_rows.sort_values(
                ["_score", "prediction_count", "_order"],
                ascending=[False, False, True],
            ).iloc[0]
            best_feature_group = best["feature_group"]
            best_metric = best["bucket_spread"] if pd.notna(best["bucket_spread"]) else best["roc_auc"]
            best_score = float(best["_score"])
            consistency = _feature_group_consistency(fg_rows)

        regime_rows = regime_comparison[
            regime_comparison.get("target_id", pd.Series(dtype=object)) == candidate.target_id
        ]
        regime_positive_count = int((regime_rows.get("direction", pd.Series(dtype=object)) == "positive").sum())
        regime_negative_count = int((regime_rows.get("direction", pd.Series(dtype=object)) == "inverted").sum())
        spreads = pd.to_numeric(regime_rows.get("bucket_spread", pd.Series(dtype=float)), errors="coerce").dropna()
        worst_regime_bucket_spread = float(spreads.min()) if not spreads.empty else pd.NA
        label, next_step, interpretation = _stability_label(
            best_score=best_score,
            feature_group_consistency=consistency,
            regime_positive_count=regime_positive_count,
            regime_negative_count=regime_negative_count,
        )
        rows.append(
            {
                "target_id": candidate.target_id,
                "display_name": candidate.display_name,
                "best_feature_group": best_feature_group,
                "best_feature_group_metric": best_metric,
                "feature_group_consistency": consistency,
                "regime_positive_count": regime_positive_count,
                "regime_negative_count": regime_negative_count,
                "worst_regime_bucket_spread": worst_regime_bucket_spread,
                "overall_stability": label,
                "next_step_candidate": next_step,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows, columns=TARGET_STABILITY_SUMMARY_COLUMNS)


def _target_row_by_id(frame: pd.DataFrame | None, target_id: object) -> pd.Series:
    if frame is None or frame.empty or "target_id" not in frame:
        return pd.Series(dtype=object)
    matches = frame[frame["target_id"] == target_id]
    if matches.empty:
        return pd.Series(dtype=object)
    return matches.iloc[0]


def _quality_summary_target_ids(
    target_balance: pd.DataFrame,
    target_walk_forward: pd.DataFrame,
    feature_group_comparison: pd.DataFrame | None,
    regime_comparison: pd.DataFrame | None,
) -> list[object]:
    ids: list[object] = []
    for frame in (target_balance, target_walk_forward, feature_group_comparison, regime_comparison):
        if frame is None or frame.empty or "target_id" not in frame:
            continue
        for target_id in frame["target_id"].dropna().tolist():
            if target_id not in ids:
                ids.append(target_id)
    return ids


def _feature_group_quality_summary(fg_rows: pd.DataFrame) -> tuple[object, object, str]:
    if fg_rows.empty:
        return pd.NA, pd.NA, "Insufficient data"

    rows = fg_rows.copy()
    rows["_score"] = rows.apply(_feature_group_score, axis=1)
    rows["_order"] = rows["feature_group"].map(
        {name: index for index, name in enumerate(DEFAULT_TARGET_FEATURE_GROUPS)}
    ).fillna(len(DEFAULT_TARGET_FEATURE_GROUPS))
    best = rows.sort_values(["_score", "prediction_count", "_order"], ascending=[False, False, True]).iloc[0]
    directions = [_feature_group_direction(row) for _, row in rows.iterrows()]
    supported = [direction for direction in directions if direction != "insufficient"]
    if not supported:
        consistency = "Insufficient data"
    elif supported.count("positive") == len(supported):
        consistency = "Stable across feature groups"
    elif "positive" in supported and (supported.count("flat") or supported.count("inverted")):
        consistency = "Feature-group dependent"
    else:
        consistency = "Weak across feature groups"
    return best.get("feature_group", pd.NA), best.get("roc_auc", pd.NA), consistency


def _regime_quality_summary(regime_rows: pd.DataFrame) -> tuple[str, object, int, int]:
    if regime_rows.empty:
        return "Insufficient data", pd.NA, 0, 0
    directions = regime_rows.get("direction", pd.Series(dtype=object)).dropna().astype(str).tolist()
    supported = [direction for direction in directions if direction != "insufficient"]
    spreads = pd.to_numeric(regime_rows.get("bucket_spread", pd.Series(dtype=float)), errors="coerce").dropna()
    worst_spread = float(spreads.min()) if not spreads.empty else pd.NA
    positive_count = supported.count("positive")
    inverted_count = supported.count("inverted")
    if not supported:
        return "Insufficient data", worst_spread, positive_count, inverted_count
    if inverted_count:
        return "Inverted in some regimes", worst_spread, positive_count, inverted_count
    if positive_count == len(supported):
        return "Stable across regimes", worst_spread, positive_count, inverted_count
    return "Regime-sensitive", worst_spread, positive_count, inverted_count


def _calibration_quality(calibration_gap: object, brier_score: object) -> str:
    gap = _numeric_value(calibration_gap)
    brier = _numeric_value(brier_score)
    if np.isnan(gap) and np.isnan(brier):
        return "Unavailable"
    abs_gap = abs(gap) if not np.isnan(gap) else 0.0
    if abs_gap <= 0.05 and (np.isnan(brier) or brier <= 0.20):
        return "Good"
    if abs_gap <= 0.10 and (np.isnan(brier) or brier <= 0.25):
        return "Acceptable"
    return "Poor"


def _bucket_separation_quality(bucket_spread: object) -> str:
    spread = _numeric_value(bucket_spread)
    if np.isnan(spread):
        return "Unavailable"
    if spread >= 0.05:
        return "Positive"
    if spread <= -0.05:
        return "Inverted"
    return "Weak"


def _overall_target_quality(
    *,
    sample_count: int,
    positive_rate: object,
    class_balance_status: object,
    overall_auc: object,
    calibration_quality: str,
    bucket_quality: str,
    feature_group_consistency: str,
    regime_stability: str,
    inverted_regime_count: int,
) -> str:
    rate = _numeric_value(positive_rate)
    auc = _numeric_value(overall_auc)
    if (
        sample_count < MIN_TARGET_DIAGNOSTIC_SAMPLE_COUNT
        or class_balance_status == "Unusable"
        or (not np.isnan(rate) and (rate < 0.05 or rate > 0.95))
    ):
        return "Unusable"
    if bucket_quality == "Inverted" or (not np.isnan(auc) and auc < 0.50) or inverted_regime_count > 1:
        return "Weak"
    if (
        class_balance_status == "Healthy"
        and bucket_quality == "Positive"
        and calibration_quality in {"Good", "Acceptable"}
        and feature_group_consistency != "Weak across feature groups"
        and regime_stability in {"Stable across regimes", "Insufficient data"}
    ):
        return "Promising"
    if calibration_quality == "Poor" or regime_stability in {"Regime-sensitive", "Inverted in some regimes"}:
        return "Mixed"
    if feature_group_consistency == "Weak across feature groups":
        return "Weak"
    return "Mixed"


def _baseline_reference(summary_rows: list[dict[str, object]]) -> dict[str, object] | None:
    for row in summary_rows:
        if row["target_id"] == "outperform_20d":
            return row
    return None


def _clearly_beats_baseline(row: dict[str, object], baseline: dict[str, object] | None) -> bool:
    if baseline is None or row["target_id"] == "outperform_20d":
        return True
    baseline_quality = str(baseline.get("overall_target_quality", ""))
    if baseline_quality in {"Weak", "Unusable"}:
        return True
    spread = _numeric_value(row.get("overall_bucket_spread"))
    baseline_spread = _numeric_value(baseline.get("overall_bucket_spread"))
    auc = _numeric_value(row.get("overall_auc"))
    baseline_auc = _numeric_value(baseline.get("overall_auc"))
    spread_better = not np.isnan(spread) and (np.isnan(baseline_spread) or spread >= baseline_spread + 0.03)
    auc_better = not np.isnan(auc) and (np.isnan(baseline_auc) or auc >= baseline_auc + 0.02)
    return spread_better or auc_better


def _production_candidate_status(row: dict[str, object], baseline: dict[str, object] | None) -> tuple[str, str]:
    target_id = str(row["target_id"])
    quality = str(row["overall_target_quality"])
    balance = str(row["class_balance_status"])
    bucket_quality = str(row["bucket_separation_quality"])
    regime_stability = str(row["regime_stability"])
    feature_consistency = str(row["feature_group_consistency"])
    if target_id == "outperform_20d":
        if quality == "Unusable":
            return "Insufficient evidence", "Keep the current target, but refresh diagnostics before using this sample."
        return "Keep baseline", "Keep the current production target until an alternative clearly beats it."
    if quality == "Unusable":
        return "Reject for now", "Do not advance this target; class balance or sample size is unusable."
    if balance == "Skewed" and bucket_quality == "Positive":
        return "Special setup only", "Review only in a special-purpose setup with explicit balance handling."
    if quality == "Promising" and _clearly_beats_baseline(row, baseline):
        if regime_stability == "Inverted in some regimes" or feature_consistency == "Weak across feature groups":
            return "Research-only candidate", "Run more segmented validation before any production trial."
        return "Candidate for production trial", "Consider a future production trial after review; do not switch automatically."
    if quality == "Weak":
        return "Reject for now", "Keep this target in diagnostics only unless future evidence improves."
    return "Research-only candidate", "Keep reviewing this target; current evidence is not enough for a production trial."


def _quality_interpretation(row: dict[str, object]) -> str:
    return (
        f"{row['overall_target_quality']} evidence: "
        f"{row['bucket_separation_quality'].lower()} bucket separation, "
        f"{row['calibration_quality'].lower()} calibration, "
        f"{row['feature_group_consistency'].lower()}, and "
        f"{row['regime_stability'].lower()}."
    )


def _quality_rank_key(row: dict[str, object]) -> tuple[float, float, float, float, str]:
    quality_order = {"Promising": 3.0, "Mixed": 2.0, "Weak": 1.0, "Unusable": 0.0}
    status_order = {
        "Candidate for production trial": 4.0,
        "Keep baseline": 3.0,
        "Special setup only": 2.0,
        "Research-only candidate": 1.5,
        "Insufficient evidence": 0.5,
        "Reject for now": 0.0,
    }
    calibration_order = {"Good": 2.0, "Acceptable": 1.0, "Unavailable": 0.5, "Poor": 0.0}
    stability_order = {
        "Stable across regimes": 2.0,
        "Regime-sensitive": 1.0,
        "Insufficient data": 0.5,
        "Inverted in some regimes": 0.0,
    }
    feature_order = {
        "Stable across feature groups": 2.0,
        "Feature-group dependent": 1.0,
        "Insufficient data": 0.5,
        "Weak across feature groups": 0.0,
    }
    spread = _numeric_value(row.get("overall_bucket_spread"))
    auc = _numeric_value(row.get("overall_auc"))
    metric_signal = 0.0
    if not np.isnan(spread):
        metric_signal += spread
    if not np.isnan(auc):
        metric_signal += auc - 0.5
    evidence_signal = (
        calibration_order.get(str(row.get("calibration_quality")), 0.0)
        + stability_order.get(str(row.get("regime_stability")), 0.0)
        + feature_order.get(str(row.get("feature_group_consistency")), 0.0)
    )
    return (
        quality_order.get(str(row.get("overall_target_quality")), 0.0),
        status_order.get(str(row.get("production_candidate_status")), 0.0),
        metric_signal,
        evidence_signal,
        str(row.get("target_id")),
    )


def build_target_quality_summary(
    target_balance: pd.DataFrame,
    target_walk_forward: pd.DataFrame,
    feature_group_comparison: pd.DataFrame | None = None,
    regime_comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify target candidates from existing diagnostics without changing production scoring.

    The rules intentionally stay transparent: first reject unusable class balance,
    then check separation, calibration, feature-group consistency, and regime stability.
    The output is a research recommendation table, not a production target selector.
    """

    rows: list[dict[str, object]] = []
    for target_id in _quality_summary_target_ids(
        target_balance,
        target_walk_forward,
        feature_group_comparison,
        regime_comparison,
    ):
        balance = _target_row_by_id(target_balance, target_id)
        walk_forward = _target_row_by_id(target_walk_forward, target_id)
        fg_rows = (
            feature_group_comparison[feature_group_comparison["target_id"] == target_id].copy()
            if feature_group_comparison is not None
            and not feature_group_comparison.empty
            and "target_id" in feature_group_comparison
            else pd.DataFrame()
        )
        regime_rows = (
            regime_comparison[regime_comparison["target_id"] == target_id].copy()
            if regime_comparison is not None and not regime_comparison.empty and "target_id" in regime_comparison
            else pd.DataFrame()
        )

        sample_count = int(_numeric_value(balance.get("sample_count")) if pd.notna(balance.get("sample_count")) else 0)
        positive_rate = balance.get("positive_rate", walk_forward.get("positive_rate", pd.NA))
        class_balance_status = balance.get("class_balance_status", "Unavailable")
        best_feature_group, best_feature_group_auc, feature_group_consistency = _feature_group_quality_summary(fg_rows)
        regime_stability, worst_regime_bucket_spread, _, inverted_regime_count = _regime_quality_summary(regime_rows)
        calibration_quality = _calibration_quality(
            walk_forward.get("calibration_gap", pd.NA),
            walk_forward.get("brier_score", pd.NA),
        )
        bucket_quality = _bucket_separation_quality(walk_forward.get("bucket_spread", pd.NA))
        overall_quality = _overall_target_quality(
            sample_count=sample_count,
            positive_rate=positive_rate,
            class_balance_status=class_balance_status,
            overall_auc=walk_forward.get("roc_auc", pd.NA),
            calibration_quality=calibration_quality,
            bucket_quality=bucket_quality,
            feature_group_consistency=feature_group_consistency,
            regime_stability=regime_stability,
            inverted_regime_count=inverted_regime_count,
        )
        rows.append(
            {
                "target_id": target_id,
                "candidate_rank": pd.NA,
                "display_name": balance.get("display_name", walk_forward.get("display_name", pd.NA)),
                "sample_count": sample_count,
                "positive_rate": positive_rate,
                "class_balance_status": class_balance_status,
                "overall_auc": walk_forward.get("roc_auc", pd.NA),
                "overall_pr_auc": walk_forward.get("pr_auc", pd.NA),
                "overall_brier_score": walk_forward.get("brier_score", pd.NA),
                "overall_calibration_gap": walk_forward.get("calibration_gap", pd.NA),
                "overall_bucket_spread": walk_forward.get("bucket_spread", pd.NA),
                "best_feature_group": best_feature_group,
                "best_feature_group_auc": best_feature_group_auc,
                "feature_group_consistency": feature_group_consistency,
                "regime_stability": regime_stability,
                "regime_inversion_count": inverted_regime_count,
                "worst_regime_bucket_spread": worst_regime_bucket_spread,
                "calibration_quality": calibration_quality,
                "bucket_separation_quality": bucket_quality,
                "overall_target_quality": overall_quality,
                "production_candidate_status": pd.NA,
                "recommended_next_step": pd.NA,
                "interpretation": pd.NA,
            }
        )

    baseline = _baseline_reference(rows)
    for row in rows:
        status, next_step = _production_candidate_status(row, baseline)
        row["production_candidate_status"] = status
        row["recommended_next_step"] = next_step
        row["interpretation"] = _quality_interpretation(row)

    ranked_rows = sorted(rows, key=_quality_rank_key, reverse=True)
    ranks = {row["target_id"]: rank for rank, row in enumerate(ranked_rows, start=1)}
    for row in rows:
        row["candidate_rank"] = ranks[row["target_id"]]
    rows = sorted(rows, key=lambda row: (row["candidate_rank"], str(row["target_id"])))
    return pd.DataFrame(rows, columns=TARGET_QUALITY_SUMMARY_COLUMNS)


def _arena_numeric(row: pd.Series, column: str) -> float:
    return _numeric_value(row.get(column))


def _arena_evidence_classification(row: pd.Series, baseline: pd.Series) -> str:
    quality = str(row.get("overall_target_quality", ""))
    class_balance = str(row.get("class_balance_status", ""))
    bucket_quality = str(row.get("bucket_separation_quality", ""))
    calibration_quality = str(row.get("calibration_quality", ""))
    regime_stability = str(row.get("regime_stability", ""))
    feature_stability = str(row.get("feature_group_consistency", ""))
    target_id = str(row.get("target_id", ""))
    sample_count = int(_arena_numeric(row, "sample_count")) if not np.isnan(_arena_numeric(row, "sample_count")) else 0
    if sample_count < MIN_TARGET_DIAGNOSTIC_SAMPLE_COUNT or quality == "Unusable" or class_balance == "Unusable":
        return "insufficient"
    if quality == "Weak" or bucket_quality == "Inverted":
        return "weak"
    if target_id == "outperform_20d":
        return "mixed" if quality == "Mixed" else "promising"

    spread = _arena_numeric(row, "overall_bucket_spread")
    baseline_spread = _arena_numeric(baseline, "overall_bucket_spread")
    brier = _arena_numeric(row, "overall_brier_score")
    baseline_brier = _arena_numeric(baseline, "overall_brier_score")
    calibration_gap = abs(_arena_numeric(row, "overall_calibration_gap"))
    baseline_gap = abs(_arena_numeric(baseline, "overall_calibration_gap"))
    baseline_inversion_value = _arena_numeric(baseline, "regime_inversion_count")
    inversion_value = _arena_numeric(row, "regime_inversion_count")
    baseline_inversions = int(baseline_inversion_value) if not np.isnan(baseline_inversion_value) else 0
    inversions = int(inversion_value) if not np.isnan(inversion_value) else 0
    spread_ok = not np.isnan(spread) and spread > 0.0
    spread_not_worse = np.isnan(baseline_spread) or spread >= baseline_spread - 0.005
    brier_not_worse = np.isnan(baseline_brier) or np.isnan(brier) or brier <= baseline_brier + 0.02
    calibration_not_worse = np.isnan(baseline_gap) or np.isnan(calibration_gap) or calibration_gap <= baseline_gap + 0.03
    regime_not_worse = inversions <= baseline_inversions
    feature_ok = feature_stability != "Weak across feature groups"
    if (
        quality == "Promising"
        and spread_ok
        and spread_not_worse
        and brier_not_worse
        and calibration_not_worse
        and regime_not_worse
        and feature_ok
        and calibration_quality in {"Good", "Acceptable"}
        and regime_stability == "Stable across regimes"
    ):
        return "strong"
    if (
        spread_ok
        and spread_not_worse
        and brier_not_worse
        and calibration_not_worse
        and regime_not_worse
        and feature_ok
        and calibration_quality != "Poor"
    ):
        return "promising"
    if quality == "Weak":
        return "weak"
    return "mixed"


def _arena_rejection_reason(row: pd.Series, classification: str) -> str:
    if classification in {"strong", "promising"}:
        return ""
    reasons = []
    if str(row.get("class_balance_status", "")) in {"Skewed", "Unusable"}:
        reasons.append(f"class balance is {str(row.get('class_balance_status')).lower()}")
    if str(row.get("calibration_quality", "")) == "Poor":
        reasons.append("calibration is poor")
    if str(row.get("bucket_separation_quality", "")) in {"Weak", "Inverted"}:
        reasons.append(f"bucket separation is {str(row.get('bucket_separation_quality')).lower()}")
    if str(row.get("regime_stability", "")) == "Inverted in some regimes":
        reasons.append("some regimes are inverted")
    elif str(row.get("regime_stability", "")) == "Regime-sensitive":
        reasons.append("evidence is regime-sensitive")
    if str(row.get("feature_group_consistency", "")) == "Weak across feature groups":
        reasons.append("feature-group stability is weak")
    return "; ".join(reasons) or "evidence is not clearly better than the current production target"


def build_target_arena_comparison(
    target_quality: pd.DataFrame,
    *,
    target_ids: tuple[str, ...] = TARGET_ARENA_TARGET_IDS,
) -> pd.DataFrame:
    """Return the research-only Target Arena v1 table.

    This is a consolidation layer over exported diagnostics. It does not select or
    change the production target.
    """

    if target_quality.empty or "target_id" not in target_quality:
        return pd.DataFrame(columns=TARGET_ARENA_COLUMNS)

    indexed = target_quality.set_index("target_id", drop=False)
    baseline = indexed.loc["outperform_20d"] if "outperform_20d" in indexed.index else pd.Series(dtype=object)
    rows: list[dict[str, object]] = []
    for target_id in target_ids:
        source_target_id = target_id
        if target_id == "drawdown_adjusted_opportunity_20d" and source_target_id not in indexed.index:
            source_target_id = "tail_adjusted_outperform_20d"
        if source_target_id not in indexed.index:
            continue
        row = indexed.loc[source_target_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if "regime_inversion_count" not in row:
            row = row.copy()
            row["regime_inversion_count"] = 0
        classification = _arena_evidence_classification(row, baseline)
        rejection_reason = _arena_rejection_reason(row, classification)
        future_experiment = classification in {"strong", "promising"} and target_id != "outperform_20d"
        if target_id == "outperform_20d":
            decision = "current_baseline"
        elif classification == "strong":
            decision = "best_candidate"
        elif classification == "promising":
            decision = "future_research_candidate"
        else:
            decision = "reject_for_now"
        rows.append(
            {
                "target_id": target_id,
                "display_name": (
                    "Drawdown-adjusted opportunity"
                    if target_id == "drawdown_adjusted_opportunity_20d"
                    else row.get("display_name", pd.NA)
                ),
                "sample_count": row.get("sample_count", pd.NA),
                "label_prevalence": row.get("positive_rate", pd.NA),
                "roc_auc": row.get("overall_auc", pd.NA),
                "pr_auc": row.get("overall_pr_auc", pd.NA),
                "brier_score": row.get("overall_brier_score", pd.NA),
                "bucket_spread": row.get("overall_bucket_spread", pd.NA),
                "calibration_gap": row.get("overall_calibration_gap", pd.NA),
                "regime_inversion_count": row.get("regime_inversion_count", 0),
                "feature_group_stability": row.get("feature_group_consistency", pd.NA),
                "evidence_classification": classification,
                "arena_decision": decision,
                "rejection_reason": rejection_reason,
                "future_production_experiment": future_experiment,
                "suggested_next_research": (
                    "Review Fourier/Wavelet signal features and regime reliability"
                    if classification in {"mixed", "weak"}
                    else "Validate in a future production-target experiment before any switch"
                ),
            }
        )
    return pd.DataFrame(rows, columns=TARGET_ARENA_COLUMNS)
