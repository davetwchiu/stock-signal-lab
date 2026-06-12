"""Diagnostics for existing out-of-sample ML signal outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml.metrics import calibration_table
from src.ml.scoring import ml_score


BASELINE_COMPARISON_COLUMNS = [
    "signal",
    "sample_size",
    "top_bucket_forward_return",
    "bottom_bucket_forward_return",
    "spread",
    "direction",
    "interpretation",
]

REGIME_SEGMENTED_COLUMNS = [
    "regime_dimension",
    "regime",
    "sample_size",
    "top_bucket_sample_size",
    "bottom_bucket_sample_size",
    "middle_bucket_forward_return",
    "top_bucket_forward_return",
    "bottom_bucket_forward_return",
    "spread",
    "direction",
    "evidence_quality",
    "interpretation",
]

DRAWDOWN_RISK_CALIBRATION_QUALITY_COLUMNS = [
    "sample_size",
    "mean_predicted_risk",
    "observed_drawdown_rate",
    "calibration_gap",
    "mean_absolute_calibration_error",
    "max_bucket_calibration_gap",
    "brier_score",
    "monotonicity",
    "interpretation",
]

LABEL_PREVALENCE_COLUMNS = [
    "label",
    "sample_size",
    "positive_count",
    "positive_rate",
    "missing_count",
    "missing_rate",
    "class_balance",
    "interpretation",
]

LABEL_THRESHOLD_SENSITIVITY_COLUMNS = [
    "target_family",
    "threshold",
    "sample_size",
    "positive_count",
    "positive_rate",
    "interpretation",
]

LABEL_DISTRIBUTION_COLUMNS = [
    "group",
    "label",
    "sample_size",
    "positive_rate",
    "class_balance",
    "interpretation",
]

LABEL_OVERLAP_COLUMNS = [
    "outperform_label",
    "drawdown_risk_label",
    "sample_size",
    "share_of_total",
    "interpretation",
]

DEFAULT_REGIME_COLUMNS = (
    "market_regime",
    "regime",
    "trend_regime",
    "risk_regime",
    "volatility_regime",
    "drawdown_regime",
)

SIMPLE_BASELINE_CANDIDATES = (
    ("return_60d", "Momentum (60d)"),
    ("return_20d", "Momentum (20d)"),
    ("rs_spy_60d", "Relative strength (SPY 60d)"),
    ("rs_qqq_60d", "Relative strength (QQQ 60d)"),
    ("dist_ma_200d", "Trend (distance from 200d MA)"),
    ("dist_ma_50d", "Trend (distance from 50d MA)"),
    ("rsi_14d", "Momentum (RSI 14d)"),
)

MIN_BASELINE_BUCKET_COUNT = 5
MIN_REGIME_BUCKET_COUNT = 5
MIN_REGIME_SAMPLE_COUNT = 20
SPREAD_SIMILARITY_TOLERANCE = 0.0025
DEFAULT_OUTPERFORMANCE_THRESHOLDS = (0.00, 0.01, 0.02, 0.03, 0.05)
DEFAULT_DRAWDOWN_THRESHOLDS = (-0.05, -0.10, -0.15, -0.20)
MIN_LABEL_AUDIT_SAMPLE_SIZE = 20
LABEL_SPARSE_RATE = 0.10
LABEL_COMMON_RATE = 0.90
LABEL_GROUP_VARIATION_THRESHOLD = 0.25


@dataclass(frozen=True)
class MLDiagnostics:
    """Diagnostic tables for the existing advisory ML signal."""

    score_buckets: pd.DataFrame
    drawdown_risk_calibration: pd.DataFrame
    summary: pd.DataFrame
    baseline_comparison: pd.DataFrame
    regime_segmented: pd.DataFrame
    drawdown_risk_calibration_quality: pd.DataFrame


@dataclass(frozen=True)
class MLLabelAudit:
    """Read-only diagnostics for existing supervised ML labels."""

    prevalence_summary: pd.DataFrame
    return_threshold_sensitivity: pd.DataFrame
    drawdown_threshold_sensitivity: pd.DataFrame
    ticker_distribution: pd.DataFrame
    regime_distribution: pd.DataFrame
    label_overlap: pd.DataFrame


def _overall_summary(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame | None,
    target: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "target": target,
        "predictions": len(predictions),
        "folds": int(predictions["fold"].nunique()) if "fold" in predictions else 0,
        "positive_rate": (
            predictions["actual"].mean() if "actual" in predictions and not predictions.empty else pd.NA
        ),
        "start": predictions["Date"].min() if "Date" in predictions and not predictions.empty else pd.NaT,
        "end": predictions["Date"].max() if "Date" in predictions and not predictions.empty else pd.NaT,
    }
    if metrics is not None and not metrics.empty:
        row.update(metrics.iloc[0].to_dict())
    return row


def _score_bucket_summary(score_panel: pd.DataFrame) -> pd.DataFrame:
    if score_panel.empty or score_panel["ML Score"].nunique() < 2:
        return pd.DataFrame()

    data = score_panel.dropna(subset=["ML Score", "actual_out"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["score_bucket"] = pd.cut(
        data["ML Score"],
        bins=[-float("inf"), 40.0, 70.0, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    aggregations = {
        "count": ("actual_out", "size"),
        "average_ml_score": ("ML Score", "mean"),
        "outperformance_hit_rate": ("actual_out", "mean"),
        "average_outperformance_probability": ("probability_out", "mean"),
        "average_drawdown_risk_probability": ("probability_risk", "mean"),
    }
    optional_columns = {
        "actual_risk": ("drawdown_risk_rate", "mean"),
        "forward_return": ("average_forward_return", "mean"),
        "forward_excess_return": ("average_forward_excess_return", "mean"),
        "forward_drawdown": ("average_forward_drawdown", "mean"),
    }
    for column, (output_name, operation) in optional_columns.items():
        if column in data:
            aggregations[output_name] = (column, operation)

    return data.groupby("score_bucket", observed=True).agg(**aggregations).reset_index()


def _empty_baseline_comparison() -> pd.DataFrame:
    return pd.DataFrame(columns=BASELINE_COMPARISON_COLUMNS)


def _empty_regime_segmented_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=REGIME_SEGMENTED_COLUMNS)


def _empty_drawdown_risk_calibration_quality() -> pd.DataFrame:
    return pd.DataFrame(columns=DRAWDOWN_RISK_CALIBRATION_QUALITY_COLUMNS)


def _empty_label_prevalence_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_PREVALENCE_COLUMNS)


def _empty_label_threshold_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_THRESHOLD_SENSITIVITY_COLUMNS)


def _empty_label_distribution() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_DISTRIBUTION_COLUMNS)


def _empty_label_overlap() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_OVERLAP_COLUMNS)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float((values * weights).sum() / weights.sum())


def _class_balance(positive_rate: object, sample_size: int) -> str:
    if sample_size < MIN_LABEL_AUDIT_SAMPLE_SIZE or pd.isna(positive_rate):
        return "insufficient"
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return "single class"
    if positive_rate < LABEL_SPARSE_RATE:
        return "sparse positive"
    if positive_rate > LABEL_COMMON_RATE:
        return "highly common positive"
    return "balanced"


def _label_balance_interpretation(positive_rate: object, sample_size: int) -> str:
    balance = _class_balance(positive_rate, sample_size)
    if balance == "insufficient":
        return "The sample is too small for reliable label audit."
    if balance == "single class":
        return "This label has a single observed class in this sample."
    if balance == "sparse positive":
        return "This label is sparse in this sample."
    if balance == "highly common positive":
        return "This label is highly common in this sample."
    return "This label has a balanced positive rate in this sample."


def _coerced_label_values(data: pd.DataFrame, label: str) -> pd.Series:
    return pd.to_numeric(data[label], errors="coerce")


def _active_label_columns(data: pd.DataFrame, horizon: int) -> list[str]:
    candidates = [
        f"label_outperform_{horizon}d",
        f"label_drawdown_risk_{horizon}d",
    ]
    return [column for column in candidates if column in data]


def build_label_prevalence_summary(
    panel: pd.DataFrame,
    *,
    horizon: int = 20,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize class balance for active supervised label columns."""

    if panel.empty:
        return _empty_label_prevalence_summary()

    labels = list(label_columns or _active_label_columns(panel, horizon))
    rows: list[dict[str, object]] = []
    total = len(panel)
    for label in labels:
        if label not in panel:
            continue
        values = _coerced_label_values(panel, label)
        sample_size = int(values.notna().sum())
        positive_count = int((values == 1).sum())
        positive_rate = float(positive_count / sample_size) if sample_size else pd.NA
        missing_count = int(total - sample_size)
        missing_rate = float(missing_count / total) if total else pd.NA
        rows.append(
            {
                "label": label,
                "sample_size": sample_size,
                "positive_count": positive_count,
                "positive_rate": positive_rate,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "class_balance": _class_balance(positive_rate, sample_size),
                "interpretation": _label_balance_interpretation(positive_rate, sample_size),
            }
        )
    if not rows:
        return _empty_label_prevalence_summary()
    return pd.DataFrame(rows, columns=LABEL_PREVALENCE_COLUMNS)


def _threshold_interpretations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return rows
    rates = pd.to_numeric(pd.Series([row["positive_rate"] for row in rows]), errors="coerce").dropna()
    high_sensitivity = not rates.empty and float(rates.max() - rates.min()) >= LABEL_GROUP_VARIATION_THRESHOLD
    interpreted: list[dict[str, object]] = []
    for row in rows:
        updated = row.copy()
        sample_size = int(updated["sample_size"])
        rate = updated["positive_rate"]
        if sample_size < MIN_LABEL_AUDIT_SAMPLE_SIZE or pd.isna(rate):
            interpretation = "The sample is too small for reliable label audit."
        elif updated["target_family"] == "drawdown_risk" and rate < LABEL_SPARSE_RATE:
            interpretation = "The drawdown-risk label is concentrated in a small number of observations."
        else:
            interpretation = _label_balance_interpretation(rate, sample_size)
        if high_sensitivity and sample_size >= MIN_LABEL_AUDIT_SAMPLE_SIZE:
            interpretation = "Threshold sensitivity is high; small threshold changes materially change class balance."
        updated["interpretation"] = interpretation
        interpreted.append(updated)
    return interpreted


def build_return_label_threshold_sensitivity(
    panel: pd.DataFrame,
    *,
    horizon: int = 20,
    thresholds: tuple[float, ...] = DEFAULT_OUTPERFORMANCE_THRESHOLDS,
) -> pd.DataFrame:
    """Audit how outperformance label prevalence changes across existing forward excess returns."""

    column = f"forward_{horizon}d_excess_return"
    if panel.empty or column not in panel:
        return _empty_label_threshold_sensitivity()
    values = pd.to_numeric(panel[column], errors="coerce").dropna()
    if values.empty:
        return _empty_label_threshold_sensitivity()

    rows: list[dict[str, object]] = []
    sample_size = int(len(values))
    for threshold in thresholds:
        positive_count = int((values > threshold).sum())
        rows.append(
            {
                "target_family": "outperformance",
                "threshold": float(threshold),
                "sample_size": sample_size,
                "positive_count": positive_count,
                "positive_rate": float(positive_count / sample_size),
                "interpretation": "",
            }
        )
    return pd.DataFrame(_threshold_interpretations(rows), columns=LABEL_THRESHOLD_SENSITIVITY_COLUMNS)


def build_drawdown_label_threshold_sensitivity(
    panel: pd.DataFrame,
    *,
    horizon: int = 20,
    thresholds: tuple[float, ...] = DEFAULT_DRAWDOWN_THRESHOLDS,
) -> pd.DataFrame:
    """Audit how drawdown-risk label prevalence changes across existing forward drawdowns."""

    column = f"forward_{horizon}d_drawdown"
    if panel.empty or column not in panel:
        return _empty_label_threshold_sensitivity()
    values = pd.to_numeric(panel[column], errors="coerce").dropna()
    if values.empty:
        return _empty_label_threshold_sensitivity()

    rows: list[dict[str, object]] = []
    sample_size = int(len(values))
    for threshold in thresholds:
        positive_count = int((values < threshold).sum())
        rows.append(
            {
                "target_family": "drawdown_risk",
                "threshold": float(threshold),
                "sample_size": sample_size,
                "positive_count": positive_count,
                "positive_rate": float(positive_count / sample_size),
                "interpretation": "",
            }
        )
    return pd.DataFrame(_threshold_interpretations(rows), columns=LABEL_THRESHOLD_SENSITIVITY_COLUMNS)


def _label_distribution_interpretation(
    group_dimension: str,
    positive_rate: object,
    sample_size: int,
    material_variation: bool,
) -> str:
    if sample_size < MIN_LABEL_AUDIT_SAMPLE_SIZE or pd.isna(positive_rate):
        return "The sample is too small for reliable label audit."
    if material_variation and group_dimension == "ticker":
        return "Positive rates vary materially by ticker."
    if material_variation and group_dimension == "regime":
        return "Positive rates vary materially by regime."
    return _label_balance_interpretation(positive_rate, sample_size)


def build_label_distribution(
    panel: pd.DataFrame,
    group_column: str,
    *,
    group_dimension: str,
    horizon: int = 20,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return label prevalence grouped by ticker or regime."""

    if panel.empty or group_column not in panel:
        return _empty_label_distribution()
    labels = list(label_columns or _active_label_columns(panel, horizon))
    if not labels:
        return _empty_label_distribution()

    data = panel.copy()
    data[group_column] = data[group_column].astype(str).str.strip()
    data = data[data[group_column] != ""]
    if data.empty:
        return _empty_label_distribution()

    rows: list[dict[str, object]] = []
    for label in labels:
        if label not in data:
            continue
        label_values = _coerced_label_values(data, label)
        rates_by_group: dict[str, float] = {}
        group_rows: list[dict[str, object]] = []
        for group, group_data in data.assign(_label=label_values).groupby(group_column, sort=True):
            values = pd.to_numeric(group_data["_label"], errors="coerce").dropna()
            sample_size = int(len(values))
            if sample_size == 0:
                continue
            positive_rate = float((values == 1).sum() / sample_size)
            rates_by_group[str(group)] = positive_rate
            group_rows.append(
                {
                    "group": str(group),
                    "label": label,
                    "sample_size": sample_size,
                    "positive_rate": positive_rate,
                    "class_balance": _class_balance(positive_rate, sample_size),
                    "interpretation": "",
                }
            )
        if not group_rows:
            continue
        rates = list(rates_by_group.values())
        material_variation = len(rates) >= 2 and max(rates) - min(rates) >= LABEL_GROUP_VARIATION_THRESHOLD
        for row in group_rows:
            row["interpretation"] = _label_distribution_interpretation(
                group_dimension,
                row["positive_rate"],
                int(row["sample_size"]),
                material_variation,
            )
            rows.append(row)
    if not rows:
        return _empty_label_distribution()
    return pd.DataFrame(rows, columns=LABEL_DISTRIBUTION_COLUMNS)


def build_return_drawdown_label_overlap(
    panel: pd.DataFrame,
    *,
    horizon: int = 20,
) -> pd.DataFrame:
    """Summarize overlap between active outperformance and drawdown-risk labels."""

    outperform_label = f"label_outperform_{horizon}d"
    drawdown_label = f"label_drawdown_risk_{horizon}d"
    if panel.empty or outperform_label not in panel or drawdown_label not in panel:
        return _empty_label_overlap()

    data = pd.DataFrame(
        {
            "outperform_label": _coerced_label_values(panel, outperform_label),
            "drawdown_risk_label": _coerced_label_values(panel, drawdown_label),
        }
    ).dropna()
    if data.empty:
        return _empty_label_overlap()
    data = data[data["outperform_label"].isin([0, 1]) & data["drawdown_risk_label"].isin([0, 1])]
    if data.empty:
        return _empty_label_overlap()

    total = len(data)
    rows: list[dict[str, object]] = []
    for (out_value, risk_value), group in data.groupby(["outperform_label", "drawdown_risk_label"], sort=True):
        sample_size = int(len(group))
        share = float(sample_size / total)
        if total < MIN_LABEL_AUDIT_SAMPLE_SIZE:
            interpretation = "The sample is too small for reliable label audit."
        elif risk_value == 1 and share < LABEL_SPARSE_RATE:
            interpretation = "The drawdown-risk label is concentrated in a small number of observations."
        else:
            interpretation = "Return-vs-drawdown label overlap is available for this sample."
        rows.append(
            {
                "outperform_label": int(out_value),
                "drawdown_risk_label": int(risk_value),
                "sample_size": sample_size,
                "share_of_total": share,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows, columns=LABEL_OVERLAP_COLUMNS)


def _first_available_regime_column(
    panel: pd.DataFrame,
    regime_cols: tuple[str, ...] = DEFAULT_REGIME_COLUMNS,
) -> str | None:
    for column in regime_cols:
        if column in panel and panel[column].notna().any():
            return column
    return None


def build_ml_label_audit(panel: pd.DataFrame, *, horizon: int = 20) -> MLLabelAudit:
    """Build compact read-only diagnostics for current supervised ML labels."""

    prevalence = build_label_prevalence_summary(panel, horizon=horizon)
    ticker_distribution = build_label_distribution(
        panel,
        "Ticker",
        group_dimension="ticker",
        horizon=horizon,
    )
    regime_column = _first_available_regime_column(panel)
    regime_distribution = (
        build_label_distribution(panel, regime_column, group_dimension="regime", horizon=horizon)
        if regime_column is not None
        else _empty_label_distribution()
    )
    return MLLabelAudit(
        prevalence,
        build_return_label_threshold_sensitivity(panel, horizon=horizon),
        build_drawdown_label_threshold_sensitivity(panel, horizon=horizon),
        ticker_distribution,
        regime_distribution,
        build_return_drawdown_label_overlap(panel, horizon=horizon),
    )


def _brier_score(
    predictions: pd.DataFrame | None,
    probability_column: str,
    label_column: str,
) -> object:
    if predictions is None or predictions.empty:
        return pd.NA
    if probability_column not in predictions or label_column not in predictions:
        return pd.NA

    data = predictions.dropna(subset=[probability_column, label_column]).copy()
    if data.empty:
        return pd.NA
    probability = pd.to_numeric(data[probability_column], errors="coerce")
    actual = pd.to_numeric(data[label_column], errors="coerce")
    usable = pd.DataFrame({"probability": probability, "actual": actual}).dropna()
    if usable.empty:
        return pd.NA
    return float(((usable["probability"] - usable["actual"]) ** 2).mean())


def _calibration_quality_interpretation(
    sample_size: int,
    calibration_gap: float,
    mean_absolute_error: float,
    max_bucket_gap: float,
    monotonicity: str,
    min_sample_size: int,
) -> str:
    if sample_size < min_sample_size:
        return "The sample is too small for reliable calibration quality assessment."
    if monotonicity == "not clearly monotonic":
        return "Higher predicted-risk buckets do not show clearly higher realised drawdown rates."
    if calibration_gap > 0.10:
        return "The model appears to underestimate realised drawdown risk in this sample."
    if calibration_gap < -0.10:
        return "The model appears to overestimate realised drawdown risk in this sample."
    if mean_absolute_error > 0.15 or max_bucket_gap > 0.25:
        return "Drawdown-risk calibration is weak in this sample."
    return "Drawdown-risk calibration looks broadly aligned in this sample."


def build_drawdown_risk_calibration_quality(
    calibration: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    probability_column: str = "probability",
    label_column: str = "actual",
    min_sample_size: int = 20,
) -> pd.DataFrame:
    """Return compact diagnostics describing drawdown-risk calibration quality."""

    required = ["count", "average_probability", "observed_drawdown_risk_rate"]
    if calibration.empty or any(column not in calibration for column in required):
        return _empty_drawdown_risk_calibration_quality()

    data = calibration[required].copy()
    data["count"] = pd.to_numeric(data["count"], errors="coerce")
    data["average_probability"] = pd.to_numeric(data["average_probability"], errors="coerce")
    data["observed_drawdown_risk_rate"] = pd.to_numeric(
        data["observed_drawdown_risk_rate"],
        errors="coerce",
    )
    data = data.dropna(subset=required)
    data = data[data["count"] > 0]
    if data.empty:
        return _empty_drawdown_risk_calibration_quality()

    sample_size = int(data["count"].sum())
    if sample_size <= 0:
        return _empty_drawdown_risk_calibration_quality()

    data = data.sort_values("average_probability")
    mean_predicted = _weighted_average(data["average_probability"], data["count"])
    observed_rate = _weighted_average(data["observed_drawdown_risk_rate"], data["count"])
    bucket_gaps = data["observed_drawdown_risk_rate"] - data["average_probability"]
    mean_absolute_error = _weighted_average(bucket_gaps.abs(), data["count"])
    max_bucket_gap = float(bucket_gaps.abs().max())
    observed_diffs = data["observed_drawdown_risk_rate"].diff().dropna()
    monotonicity = (
        "higher buckets aligned"
        if len(data) >= 2 and bool((observed_diffs >= -1e-12).all())
        else "not clearly monotonic"
    )
    calibration_gap = float(observed_rate - mean_predicted)
    interpretation = _calibration_quality_interpretation(
        sample_size,
        calibration_gap,
        mean_absolute_error,
        max_bucket_gap,
        monotonicity,
        min_sample_size,
    )

    return pd.DataFrame(
        [
            {
                "sample_size": sample_size,
                "mean_predicted_risk": mean_predicted,
                "observed_drawdown_rate": observed_rate,
                "calibration_gap": calibration_gap,
                "mean_absolute_calibration_error": mean_absolute_error,
                "max_bucket_calibration_gap": max_bucket_gap,
                "brier_score": _brier_score(predictions, probability_column, label_column),
                "monotonicity": monotonicity,
                "interpretation": interpretation,
            }
        ],
        columns=DRAWDOWN_RISK_CALIBRATION_QUALITY_COLUMNS,
    )


def _comparison_direction(spread: object) -> str:
    if pd.isna(spread):
        return "insufficient"
    if spread > 0:
        return "top bucket higher"
    if spread < 0:
        return "bottom bucket higher"
    return "flat"


def _comparison_row(
    signal: str,
    sample_size: int,
    top_return: float | None,
    bottom_return: float | None,
) -> dict[str, object]:
    spread = pd.NA
    if top_return is not None and bottom_return is not None:
        spread = float(top_return - bottom_return)
    return {
        "signal": signal,
        "sample_size": int(sample_size),
        "top_bucket_forward_return": pd.NA if top_return is None else float(top_return),
        "bottom_bucket_forward_return": pd.NA if bottom_return is None else float(bottom_return),
        "spread": spread,
        "direction": _comparison_direction(spread),
        "interpretation": "The sample is too small for a reliable baseline comparison."
        if pd.isna(spread)
        else "",
    }


def _preferred_simple_baseline(data: pd.DataFrame) -> tuple[str, str] | None:
    for column, label in SIMPLE_BASELINE_CANDIDATES:
        if column in data and data[column].notna().sum() >= MIN_BASELINE_BUCKET_COUNT * 2:
            return column, label
    return None


def _merge_baseline_panel(score_panel: pd.DataFrame, baseline_panel: pd.DataFrame | None) -> pd.DataFrame:
    if baseline_panel is None or baseline_panel.empty:
        return score_panel.copy()
    if "Date" not in score_panel or "Ticker" not in score_panel:
        return score_panel.copy()
    if "Date" not in baseline_panel or "Ticker" not in baseline_panel:
        return score_panel.copy()

    candidate_columns = [column for column, _ in SIMPLE_BASELINE_CANDIDATES if column in baseline_panel]
    if not candidate_columns:
        return score_panel.copy()

    left = score_panel.copy()
    right = baseline_panel[["Date", "Ticker", *candidate_columns]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])
    right = right.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return left.merge(right, on=["Date", "Ticker"], how="left")


def _ml_score_comparison_row(
    data: pd.DataFrame,
    forward_return_column: str,
    min_bucket_size: int,
) -> dict[str, object]:
    if "ML Score" not in data or forward_return_column not in data:
        return _comparison_row("ML score", 0, None, None)

    usable = data.dropna(subset=["ML Score", forward_return_column]).copy()
    if usable.empty or len(usable) < min_bucket_size * 2:
        return _comparison_row("ML score", len(usable), None, None)

    usable["score_bucket"] = pd.cut(
        usable["ML Score"],
        bins=[-float("inf"), 40.0, 70.0, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    low = usable[usable["score_bucket"] == "Low"]
    high = usable[usable["score_bucket"] == "High"]
    if len(low) < min_bucket_size or len(high) < min_bucket_size:
        return _comparison_row("ML score", len(usable), None, None)
    return _comparison_row(
        "ML score",
        len(usable),
        high[forward_return_column].mean(),
        low[forward_return_column].mean(),
    )


def _no_skill_comparison_row(
    data: pd.DataFrame,
    forward_return_column: str,
    min_bucket_size: int,
) -> dict[str, object]:
    if forward_return_column not in data:
        return _comparison_row("No-skill / universe average", 0, None, None)
    usable = data.dropna(subset=[forward_return_column])
    if len(usable) < min_bucket_size * 2:
        return _comparison_row("No-skill / universe average", len(usable), None, None)
    average_return = float(usable[forward_return_column].mean())
    return _comparison_row(
        "No-skill / universe average",
        len(usable),
        average_return,
        average_return,
    )


def _simple_baseline_comparison_row(
    data: pd.DataFrame,
    baseline_column: str,
    signal: str,
    forward_return_column: str,
    min_bucket_size: int,
) -> dict[str, object]:
    if baseline_column not in data or forward_return_column not in data:
        return _comparison_row(signal, 0, None, None)

    usable = data.dropna(subset=[baseline_column, forward_return_column]).copy()
    if len(usable) < min_bucket_size * 2 or usable[baseline_column].nunique() < 2:
        return _comparison_row(signal, len(usable), None, None)

    try:
        usable["baseline_bucket"] = pd.qcut(
            usable[baseline_column],
            q=3,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return _comparison_row(signal, len(usable), None, None)

    top_bucket = usable["baseline_bucket"].max()
    bottom_bucket = usable["baseline_bucket"].min()
    if pd.isna(top_bucket) or pd.isna(bottom_bucket) or top_bucket == bottom_bucket:
        return _comparison_row(signal, len(usable), None, None)

    bottom = usable[usable["baseline_bucket"] == bottom_bucket]
    top = usable[usable["baseline_bucket"] == top_bucket]
    if len(bottom) < min_bucket_size or len(top) < min_bucket_size:
        return _comparison_row(signal, len(usable), None, None)
    return _comparison_row(
        signal,
        len(usable),
        top[forward_return_column].mean(),
        bottom[forward_return_column].mean(),
    )


def _spread_value(row: pd.Series) -> float | None:
    value = pd.to_numeric(pd.Series([row.get("spread")]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def _add_baseline_interpretations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows

    ml_rows = frame[frame["signal"] == "ML score"]
    ml_spread = None if ml_rows.empty else _spread_value(ml_rows.iloc[0])
    baseline_spreads = [
        _spread_value(row)
        for _, row in frame[frame["signal"] != "ML score"].iterrows()
    ]
    usable_baseline_spreads = [value for value in baseline_spreads if value is not None]
    best_baseline_spread = max(usable_baseline_spreads) if usable_baseline_spreads else 0.0

    interpreted: list[dict[str, object]] = []
    for row in rows:
        updated = row.copy()
        spread = pd.to_numeric(pd.Series([updated.get("spread")]), errors="coerce").iloc[0]
        if pd.isna(spread):
            updated["interpretation"] = "The sample is too small for a reliable baseline comparison."
        elif updated["signal"] == "No-skill / universe average":
            updated["interpretation"] = (
                "No-skill reference uses the same average forward return for top and bottom buckets."
            )
        elif updated["signal"] == "ML score":
            if ml_spread is None:
                updated["interpretation"] = "The sample is too small for a reliable baseline comparison."
            elif ml_spread > best_baseline_spread + SPREAD_SIMILARITY_TOLERANCE:
                updated["interpretation"] = (
                    "ML score shows better bucket spread than the simple baseline in this sample."
                )
            elif ml_spread + SPREAD_SIMILARITY_TOLERANCE < best_baseline_spread:
                updated["interpretation"] = (
                    "ML score does not improve on the simple baseline. Treat ML evidence cautiously."
                )
            else:
                updated["interpretation"] = (
                    "The baseline and ML score are similar. This suggests the ML score may mostly be "
                    "capturing simple trend or momentum."
                )
        elif ml_spread is None:
            updated["interpretation"] = "The sample is too small for a reliable baseline comparison."
        elif float(spread) > ml_spread + SPREAD_SIMILARITY_TOLERANCE:
            updated["interpretation"] = (
                "This simple baseline shows better bucket spread than ML score in this sample."
            )
        elif float(spread) + SPREAD_SIMILARITY_TOLERANCE < ml_spread:
            updated["interpretation"] = (
                "ML score shows better bucket spread than this simple baseline in this sample."
            )
        else:
            updated["interpretation"] = (
                "The baseline and ML score are similar. This suggests the ML score may mostly be "
                "capturing simple trend or momentum."
            )
        interpreted.append(updated)
    return interpreted


def build_ml_baseline_comparison(
    score_panel: pd.DataFrame,
    baseline_panel: pd.DataFrame | None = None,
    forward_return_column: str = "forward_return",
    min_bucket_size: int = MIN_BASELINE_BUCKET_COUNT,
) -> pd.DataFrame:
    """Return a compact DataFrame comparing ML score bucket separation with simple baselines."""

    if score_panel.empty:
        return _empty_baseline_comparison()

    data = _merge_baseline_panel(score_panel, baseline_panel)
    rows = [
        _ml_score_comparison_row(data, forward_return_column, min_bucket_size),
        _no_skill_comparison_row(data, forward_return_column, min_bucket_size),
    ]
    simple_baseline = _preferred_simple_baseline(data)
    if simple_baseline is not None:
        baseline_column, signal = simple_baseline
        rows.append(
            _simple_baseline_comparison_row(
                data,
                baseline_column,
                signal,
                forward_return_column,
                min_bucket_size,
            )
        )

    return pd.DataFrame(_add_baseline_interpretations(rows), columns=BASELINE_COMPARISON_COLUMNS)


def _merge_regime_panel(
    score_panel: pd.DataFrame,
    baseline_panel: pd.DataFrame | None,
    regime_columns: list[str],
) -> pd.DataFrame:
    data = score_panel.copy()
    missing_columns = [column for column in regime_columns if column not in data]
    if not missing_columns:
        return data
    if baseline_panel is None or baseline_panel.empty:
        return data
    if "Date" not in data or "Ticker" not in data:
        return data
    if "Date" not in baseline_panel or "Ticker" not in baseline_panel:
        return data

    available_columns = [column for column in missing_columns if column in baseline_panel]
    if not available_columns:
        return data

    right = baseline_panel[["Date", "Ticker", *available_columns]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    right["Date"] = pd.to_datetime(right["Date"])
    right = right.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return data.merge(right, on=["Date", "Ticker"], how="left")


def _regime_interpretation(
    spread: object,
    sample_size: int,
    *,
    min_samples: int,
    concentrated: bool,
) -> tuple[str, str]:
    if sample_size < min_samples or pd.isna(spread):
        return (
            "insufficient",
            "The regime sample is too small for reliable interpretation.",
        )

    concentration_note = (
        " Regime comparison is limited because the sample is concentrated in one regime."
        if concentrated
        else ""
    )
    if spread > SPREAD_SIMILARITY_TOLERANCE:
        return (
            "usable",
            "ML score shows positive separation in this regime." + concentration_note,
        )
    if spread < -SPREAD_SIMILARITY_TOLERANCE:
        return (
            "inverted",
            "ML score separation is inverted in this regime." + concentration_note,
        )
    return (
        "mixed",
        "ML score separation is weak or inconclusive in this regime." + concentration_note,
    )


def _regime_segment_row(
    data: pd.DataFrame,
    *,
    regime_dimension: str,
    regime: object,
    score_col: str,
    forward_return_col: str,
    min_bucket_size: int,
    min_samples: int,
    concentrated: bool,
) -> dict[str, object]:
    sample_size = int(len(data))
    row = {
        "regime_dimension": regime_dimension,
        "regime": regime,
        "sample_size": sample_size,
        "top_bucket_sample_size": 0,
        "bottom_bucket_sample_size": 0,
        "middle_bucket_forward_return": pd.NA,
        "top_bucket_forward_return": pd.NA,
        "bottom_bucket_forward_return": pd.NA,
        "spread": pd.NA,
        "direction": "insufficient",
        "evidence_quality": "insufficient",
        "interpretation": "The regime sample is too small for reliable interpretation.",
    }
    if sample_size < min_samples or data[score_col].nunique() < 2:
        return row

    bucketed = data.copy()
    bucketed["score_bucket"] = pd.cut(
        bucketed[score_col],
        bins=[-float("inf"), 40.0, 70.0, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    low = bucketed[bucketed["score_bucket"] == "Low"]
    medium = bucketed[bucketed["score_bucket"] == "Medium"]
    high = bucketed[bucketed["score_bucket"] == "High"]
    row["top_bucket_sample_size"] = int(len(high))
    row["bottom_bucket_sample_size"] = int(len(low))
    if not medium.empty:
        row["middle_bucket_forward_return"] = float(medium[forward_return_col].mean())
    if len(low) < min_bucket_size or len(high) < min_bucket_size:
        return row

    top_return = float(high[forward_return_col].mean())
    bottom_return = float(low[forward_return_col].mean())
    spread = top_return - bottom_return
    evidence_quality, interpretation = _regime_interpretation(
        spread,
        sample_size,
        min_samples=min_samples,
        concentrated=concentrated,
    )
    row.update(
        {
            "top_bucket_forward_return": top_return,
            "bottom_bucket_forward_return": bottom_return,
            "spread": spread,
            "direction": _comparison_direction(spread),
            "evidence_quality": evidence_quality,
            "interpretation": interpretation,
        }
    )
    return row


def build_regime_segmented_ml_diagnostics(
    score_panel: pd.DataFrame,
    *,
    baseline_panel: pd.DataFrame | None = None,
    score_col: str = "ML Score",
    forward_return_col: str = "forward_return",
    regime_cols: list[str] | None = None,
    min_bucket_size: int = MIN_REGIME_BUCKET_COUNT,
    min_samples: int = MIN_REGIME_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Return ML score bucket separation grouped by existing regime columns."""

    if score_panel.empty:
        return _empty_regime_segmented_diagnostics()

    candidate_regime_cols = list(regime_cols or DEFAULT_REGIME_COLUMNS)
    data = _merge_regime_panel(score_panel, baseline_panel, candidate_regime_cols)
    required = [score_col, forward_return_col]
    if any(column not in data for column in required):
        return _empty_regime_segmented_diagnostics()

    available_regime_cols = [column for column in candidate_regime_cols if column in data]
    if not available_regime_cols:
        return _empty_regime_segmented_diagnostics()

    usable = data.dropna(subset=[score_col, forward_return_col]).copy()
    if usable.empty:
        return _empty_regime_segmented_diagnostics()

    rows: list[dict[str, object]] = []
    for regime_col in available_regime_cols:
        regime_data = usable.dropna(subset=[regime_col]).copy()
        if regime_data.empty:
            continue
        regime_data[regime_col] = regime_data[regime_col].astype(str).str.strip()
        regime_data = regime_data[regime_data[regime_col] != ""]
        if regime_data.empty:
            continue

        concentrated = regime_data[regime_col].nunique() <= 1
        for regime, group in regime_data.groupby(regime_col, sort=True):
            rows.append(
                _regime_segment_row(
                    group,
                    regime_dimension=regime_col,
                    regime=regime,
                    score_col=score_col,
                    forward_return_col=forward_return_col,
                    min_bucket_size=min_bucket_size,
                    min_samples=min_samples,
                    concentrated=concentrated,
                )
            )

    return pd.DataFrame(rows, columns=REGIME_SEGMENTED_COLUMNS)


def build_ml_diagnostics(
    outperformance_predictions: pd.DataFrame,
    drawdown_risk_predictions: pd.DataFrame,
    outperformance_metrics: pd.DataFrame | None = None,
    drawdown_risk_metrics: pd.DataFrame | None = None,
    risk_bins: int = 5,
    baseline_panel: pd.DataFrame | None = None,
) -> MLDiagnostics:
    """Summarize out-of-sample usefulness of the existing ML signal.

    Inputs should come from existing walk-forward validation results. This helper
    only joins predictions and summarizes diagnostics; it does not fit models or
    change score, probability, or decision logic.
    """

    if outperformance_predictions.empty or drawdown_risk_predictions.empty:
        summary = pd.DataFrame(
            [
                _overall_summary(outperformance_predictions, outperformance_metrics, "outperformance"),
                _overall_summary(drawdown_risk_predictions, drawdown_risk_metrics, "drawdown_risk"),
            ]
        )
        return MLDiagnostics(
            pd.DataFrame(),
            pd.DataFrame(),
            summary,
            _empty_baseline_comparison(),
            _empty_regime_segmented_diagnostics(),
            _empty_drawdown_risk_calibration_quality(),
        )

    out_columns = [
        "Date",
        "Ticker",
        "actual",
        "probability",
        "forward_return",
        "forward_excess_return",
        "forward_drawdown",
    ]
    risk_columns = ["Date", "Ticker", "actual", "probability"]
    out = outperformance_predictions[
        [column for column in out_columns if column in outperformance_predictions]
    ].rename(columns={"actual": "actual_out", "probability": "probability_out"})
    risk = drawdown_risk_predictions[
        [column for column in risk_columns if column in drawdown_risk_predictions]
    ].rename(
        columns={"actual": "actual_risk", "probability": "probability_risk"}
    )
    score_panel = out.merge(risk, on=["Date", "Ticker"], how="inner")
    if not score_panel.empty:
        score_panel["ML Score"] = ml_score(score_panel["probability_out"], score_panel["probability_risk"])
    baseline_comparison = build_ml_baseline_comparison(score_panel, baseline_panel)
    regime_segmented = build_regime_segmented_ml_diagnostics(score_panel, baseline_panel=baseline_panel)

    risk_calibration = calibration_table(
        drawdown_risk_predictions,
        probability_column="probability",
        label_column="actual",
        bins=risk_bins,
    ).rename(columns={"observed_rate": "observed_drawdown_risk_rate"})
    risk_calibration_quality = build_drawdown_risk_calibration_quality(
        risk_calibration,
        drawdown_risk_predictions,
    )
    summary = pd.DataFrame(
        [
            _overall_summary(outperformance_predictions, outperformance_metrics, "outperformance"),
            _overall_summary(drawdown_risk_predictions, drawdown_risk_metrics, "drawdown_risk"),
        ]
    )
    return MLDiagnostics(
        _score_bucket_summary(score_panel),
        risk_calibration,
        summary,
        baseline_comparison,
        regime_segmented,
        risk_calibration_quality,
    )
