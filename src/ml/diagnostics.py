"""Diagnostics for existing out-of-sample ML signal outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml.datasets import feature_group_columns
from src.ml.metrics import calibration_table
from src.ml.scoring import ml_score
from src.ml.validation import deduplicate_prediction_keys, prediction_merge_keys


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

SCORE_DIRECTION_SUMMARY_COLUMNS = [
    "sample_size",
    "score_column",
    "target_column",
    "label_column",
    "drawdown_label_column",
    "score_to_forward_return_spearman",
    "score_to_return_label_spearman",
    "score_to_drawdown_label_spearman",
    "top_bucket_forward_return",
    "bottom_bucket_forward_return",
    "top_minus_bottom_spread",
    "direction",
    "interpretation",
]

PROBABILITY_LABEL_ALIGNMENT_COLUMNS = [
    "diagnostic",
    "sample_size",
    "bottom_score_bucket_mean",
    "top_score_bucket_mean",
    "top_minus_bottom_spread",
    "higher_score_corresponds_to",
    "interpretation",
]

SCORE_BUCKET_MONOTONICITY_COLUMNS = [
    "bucket",
    "sample_size",
    "mean_score",
    "mean_forward_return",
    "return_label_rate",
    "drawdown_label_rate",
    "monotonicity_result",
    "interpretation",
]

SCORE_INVERSION_COLUMNS = [
    "score_direction",
    "sample_size",
    "top_bucket_forward_return",
    "bottom_bucket_forward_return",
    "top_minus_bottom_spread",
    "better_forward_return_separation",
    "interpretation",
]

ML_PROBABILITY_DIRECTION_COLUMNS = [
    "signal",
    "sample_size",
    "low_bucket_forward_excess_return",
    "mid_bucket_forward_excess_return",
    "high_bucket_forward_excess_return",
    "high_minus_low_spread",
    "monotonicity",
    "actual_label_rate_low_bucket",
    "actual_label_rate_high_bucket",
    "interpretation",
]

ML_FORMULA_CANDIDATE_COLUMNS = [
    "candidate_name",
    "sample_size",
    "low_bucket_forward_excess_return",
    "mid_bucket_forward_excess_return",
    "high_bucket_forward_excess_return",
    "high_minus_low_spread",
    "monotonicity",
    "low_bucket_label_rate",
    "high_bucket_label_rate",
    "low_bucket_drawdown_event_rate",
    "high_bucket_drawdown_event_rate",
    "interpretation",
]

REGIME_SCORE_DIRECTION_COLUMNS = [
    "regime_dimension",
    "regime",
    "sample_size",
    "top_bucket_forward_return",
    "bottom_bucket_forward_return",
    "top_minus_bottom_spread",
    "direction",
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

TARGET_COMPARISON_COLUMNS = [
    "target_version",
    "target_column",
    "sample_size",
    "low_bucket_target",
    "middle_bucket_target",
    "high_bucket_target",
    "high_minus_low_spread",
    "monotonicity",
    "baseline_spread",
    "relative_result",
    "interpretation",
]

OPPORTUNITY_RISK_JOINT_COLUMNS = [
    "opportunity_bucket",
    "risk_bucket",
    "sample_size",
    "avg_forward_excess_return",
    "avg_forward_drawdown",
    "drawdown_event_rate",
    "interpretation",
    "joint_validation_result",
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
MIN_SCORE_DIRECTION_BUCKET_COUNT = 5
MIN_SCORE_DIRECTION_SAMPLE_COUNT = 20
MIN_JOINT_VALIDATION_CELL_COUNT = 5
MIN_JOINT_VALIDATION_SAMPLE_COUNT = 20
DEFAULT_OUTPERFORMANCE_THRESHOLDS = (0.00, 0.01, 0.02, 0.03, 0.05)
DEFAULT_DRAWDOWN_THRESHOLDS = (-0.05, -0.10, -0.15, -0.20)
MIN_LABEL_AUDIT_SAMPLE_SIZE = 20
LABEL_SPARSE_RATE = 0.10
LABEL_COMMON_RATE = 0.90
LABEL_GROUP_VARIATION_THRESHOLD = 0.25
HIGH_MISSING_RATE = 0.30
NEAR_CONSTANT_DOMINANCE = 0.95
LOW_SAMPLE_TO_FEATURE_RATIO = 10.0
COMPLEX_FEATURE_SAMPLE_RATIO = 20.0
FAMILY_DOMINANCE_SHARE = 0.60
HIGH_CORRELATION_THRESHOLD = 0.95
MAX_CORRELATION_PAIRS = 10
MIN_FEATURE_SIGNAL_SAMPLE_SIZE = 20
MIN_FEATURE_SIGNAL_UNIQUE_VALUES = 3
FEATURE_SIGNAL_TOP_N = 15
FEATURE_SIGNAL_QUANTILE_COUNT = 5
NEAR_ZERO_FEATURE_SIGNAL = 0.05
COMPLEX_TOP_SIGNAL_SHARE = 0.50

FEATURE_INVENTORY_COLUMNS = [
    "feature_count",
    "numeric_feature_count",
    "non_numeric_feature_count",
    "missing_feature_count",
    "sample_size",
    "sample_to_feature_ratio",
    "mean_missing_rate",
    "high_missing_feature_count",
    "constant_or_near_constant_feature_count",
    "interpretation",
]

FEATURE_FAMILY_COLUMNS = [
    "family",
    "feature_count",
    "share_of_features",
    "example_features",
    "interpretation",
]

FEATURE_WARNING_COLUMNS = [
    "warning",
    "severity",
    "feature_count",
    "detail",
    "interpretation",
]

FEATURE_REDUNDANCY_SUMMARY_COLUMNS = [
    "numeric_feature_count",
    "high_correlation_pair_count",
    "correlation_threshold",
    "interpretation",
]

FEATURE_CORRELATION_PAIR_COLUMNS = [
    "feature_1",
    "feature_2",
    "abs_correlation",
    "interpretation",
]

FEATURE_IMPORTANCE_COLUMNS = [
    "feature",
    "importance",
    "rank",
    "source",
    "interpretation",
]

FEATURE_SIGNAL_COLUMNS = [
    "feature",
    "family",
    "valid_sample_count",
    "missing_rate",
    "unique_value_count",
    "abs_spearman_to_return_target",
    "spearman_to_return_target",
    "abs_spearman_to_return_label",
    "abs_spearman_to_drawdown_label",
    "top_quantile_target_mean",
    "bottom_quantile_target_mean",
    "quantile_spread",
    "interpretation",
]

FEATURE_FAMILY_SIGNAL_COLUMNS = [
    "family",
    "feature_count",
    "median_abs_signal",
    "max_abs_signal",
    "top_feature",
    "top_feature_signal",
    "share_of_top_features",
    "interpretation",
]

FEATURE_QUANTILE_SIGNAL_COLUMNS = [
    "feature",
    "family",
    "valid_sample_count",
    "bottom_quantile_target_mean",
    "top_quantile_target_mean",
    "quantile_spread",
    "abs_quantile_spread",
    "interpretation",
]

FEATURE_FAMILY_KEYWORDS = (
    ("momentum / return", ("return", "momentum", "roc")),
    ("trend / moving average", ("ma_", "moving_average", "sma", "ema", "trend", "dist_ma")),
    ("volatility", ("volatility", "atr", "drawdown")),
    ("relative strength / benchmark-relative", ("rs_", "relative_strength", "benchmark", "excess")),
    ("volume / liquidity", ("volume", "liquidity", "turnover")),
    ("RSI / technical", ("rsi", "macd", "stoch", "technical")),
    ("Fourier / wavelet / complex transform", ("fourier", "wavelet")),
)


@dataclass(frozen=True)
class MLDiagnostics:
    """Diagnostic tables for the existing advisory ML signal."""

    score_buckets: pd.DataFrame
    drawdown_risk_calibration: pd.DataFrame
    summary: pd.DataFrame
    baseline_comparison: pd.DataFrame
    regime_segmented: pd.DataFrame
    score_direction_summary: pd.DataFrame
    probability_label_alignment: pd.DataFrame
    score_bucket_monotonicity: pd.DataFrame
    score_inversion: pd.DataFrame
    regime_score_direction: pd.DataFrame
    drawdown_risk_calibration_quality: pd.DataFrame
    target_comparison: pd.DataFrame
    opportunity_risk_joint_validation: pd.DataFrame
    probability_direction_check: pd.DataFrame
    formula_candidate_comparison: pd.DataFrame


@dataclass(frozen=True)
class MLLabelAudit:
    """Read-only diagnostics for existing supervised ML labels."""

    prevalence_summary: pd.DataFrame
    return_threshold_sensitivity: pd.DataFrame
    drawdown_threshold_sensitivity: pd.DataFrame
    ticker_distribution: pd.DataFrame
    regime_distribution: pd.DataFrame
    label_overlap: pd.DataFrame


@dataclass(frozen=True)
class MLFeatureAudit:
    """Read-only diagnostics for the current ML feature set."""

    inventory_summary: pd.DataFrame
    family_summary: pd.DataFrame
    warnings: pd.DataFrame
    redundancy_summary: pd.DataFrame
    high_correlation_pairs: pd.DataFrame
    feature_importance: pd.DataFrame


@dataclass(frozen=True)
class MLFeatureSignalDiagnostics:
    """Read-only univariate signal diagnostics for the current ML feature set."""

    signal_table: pd.DataFrame
    family_summary: pd.DataFrame
    quantile_summary: pd.DataFrame
    warnings: pd.DataFrame


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


def _empty_score_direction_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=SCORE_DIRECTION_SUMMARY_COLUMNS)


def _empty_probability_label_alignment() -> pd.DataFrame:
    return pd.DataFrame(columns=PROBABILITY_LABEL_ALIGNMENT_COLUMNS)


def _empty_score_bucket_monotonicity() -> pd.DataFrame:
    return pd.DataFrame(columns=SCORE_BUCKET_MONOTONICITY_COLUMNS)


def _empty_score_inversion_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=SCORE_INVERSION_COLUMNS)


def _empty_ml_probability_direction_check() -> pd.DataFrame:
    return pd.DataFrame(columns=ML_PROBABILITY_DIRECTION_COLUMNS)


def _empty_ml_formula_candidate_comparison() -> pd.DataFrame:
    return pd.DataFrame(columns=ML_FORMULA_CANDIDATE_COLUMNS)


def _empty_regime_score_direction_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=REGIME_SCORE_DIRECTION_COLUMNS)


def _empty_drawdown_risk_calibration_quality() -> pd.DataFrame:
    return pd.DataFrame(columns=DRAWDOWN_RISK_CALIBRATION_QUALITY_COLUMNS)


def _empty_target_comparison() -> pd.DataFrame:
    return pd.DataFrame(columns=TARGET_COMPARISON_COLUMNS)


def _empty_opportunity_risk_joint_validation() -> pd.DataFrame:
    return pd.DataFrame(columns=OPPORTUNITY_RISK_JOINT_COLUMNS)


def _empty_label_prevalence_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_PREVALENCE_COLUMNS)


def _empty_label_threshold_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_THRESHOLD_SENSITIVITY_COLUMNS)


def _empty_label_distribution() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_DISTRIBUTION_COLUMNS)


def _empty_label_overlap() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_OVERLAP_COLUMNS)


def _empty_feature_inventory_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_INVENTORY_COLUMNS)


def _empty_feature_family_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_FAMILY_COLUMNS)


def _empty_feature_warnings() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_WARNING_COLUMNS)


def _empty_feature_redundancy_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_REDUNDANCY_SUMMARY_COLUMNS)


def _empty_feature_correlation_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_CORRELATION_PAIR_COLUMNS)


def _empty_feature_importance_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)


def _empty_feature_signal_table() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_SIGNAL_COLUMNS)


def _empty_feature_family_signal_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_FAMILY_SIGNAL_COLUMNS)


def _empty_feature_quantile_signal_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_QUANTILE_SIGNAL_COLUMNS)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float((values * weights).sum() / weights.sum())


def _coerce_feature_columns(panel: pd.DataFrame, feature_columns: list[str] | None) -> list[str]:
    if feature_columns is not None:
        return list(dict.fromkeys(feature_columns))

    return feature_group_columns(panel, "all")


def _feature_inventory_interpretation(
    sample_size: int,
    feature_count: int,
    sample_to_feature_ratio: object,
    high_missing_count: int,
    constant_count: int,
) -> str:
    if sample_size == 0 or feature_count == 0:
        return "Feature audit is unavailable because the sample or feature list is empty."
    if high_missing_count or constant_count:
        return "Feature set has missingness or low-variation fields that may weaken validation evidence."
    if pd.notna(sample_to_feature_ratio) and sample_to_feature_ratio < LOW_SAMPLE_TO_FEATURE_RATIO:
        return "Feature set is wide relative to the sample, so validation evidence may be fragile."
    return "Feature inventory looks usable for research diagnostics in this sample."


def _feature_family(feature: str) -> str:
    normalized = feature.lower()
    for family, keywords in FEATURE_FAMILY_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return family
    return "unknown / other"


def _numeric_audit_features(panel: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    numeric_features: list[str] = []
    for column in feature_columns:
        if column not in panel:
            continue
        if pd.api.types.is_numeric_dtype(panel[column]):
            numeric_features.append(column)
            continue
        coerced = pd.to_numeric(panel[column], errors="coerce")
        if coerced.notna().any() or panel[column].isna().all():
            numeric_features.append(column)
    return numeric_features


def build_feature_family_summary(feature_columns: list[str]) -> pd.DataFrame:
    """Group feature names into simple diagnostic families."""

    features = list(dict.fromkeys(feature_columns))
    if not features:
        return _empty_feature_family_summary()

    rows: list[dict[str, object]] = []
    total = len(features)
    family_order = [family for family, _ in FEATURE_FAMILY_KEYWORDS] + ["unknown / other"]
    for family in family_order:
        family_features = [feature for feature in features if _feature_family(feature) == family]
        if not family_features:
            continue
        share = float(len(family_features) / total)
        rows.append(
            {
                "family": family,
                "feature_count": len(family_features),
                "share_of_features": share,
                "example_features": ", ".join(family_features[:5]),
                "interpretation": "This family dominates the feature set."
                if share >= FAMILY_DOMINANCE_SHARE
                else "This family is represented in the feature set.",
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_FAMILY_COLUMNS)


def _near_constant_features(data: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for column in data.columns:
        values = data[column].dropna()
        if values.empty:
            features.append(column)
            continue
        if values.nunique(dropna=True) <= 1:
            features.append(column)
            continue
        top_share = float(values.value_counts(normalize=True, dropna=True).iloc[0])
        if top_share >= NEAR_CONSTANT_DOMINANCE:
            features.append(column)
    return features


def build_feature_inventory_summary(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize feature count, sample size, missingness, and low-variation fields."""

    if panel.empty:
        return pd.DataFrame(
            [
                {
                    "feature_count": 0,
                    "numeric_feature_count": 0,
                    "non_numeric_feature_count": 0,
                    "missing_feature_count": 0,
                    "sample_size": 0,
                    "sample_to_feature_ratio": pd.NA,
                    "mean_missing_rate": pd.NA,
                    "high_missing_feature_count": 0,
                    "constant_or_near_constant_feature_count": 0,
                    "interpretation": _feature_inventory_interpretation(0, 0, pd.NA, 0, 0),
                }
            ],
            columns=FEATURE_INVENTORY_COLUMNS,
        )

    features = _coerce_feature_columns(panel, feature_columns)
    present_features = [column for column in features if column in panel]
    missing_feature_count = len([column for column in features if column not in panel])
    numeric_features = _numeric_audit_features(panel, present_features)
    non_numeric_count = len(present_features) - len(numeric_features)
    numeric_data = (
        panel[numeric_features].apply(pd.to_numeric, errors="coerce")
        if numeric_features
        else pd.DataFrame()
    )
    missing_rates = numeric_data.isna().mean() if not numeric_data.empty else pd.Series(dtype=float)
    high_missing_count = int((missing_rates >= HIGH_MISSING_RATE).sum()) if not missing_rates.empty else 0
    near_constant_count = len(_near_constant_features(numeric_data)) if not numeric_data.empty else 0
    feature_count = len(features)
    sample_size = len(panel)
    sample_to_feature_ratio = float(sample_size / feature_count) if feature_count else pd.NA
    mean_missing_rate = float(missing_rates.mean()) if not missing_rates.empty else pd.NA

    return pd.DataFrame(
        [
            {
                "feature_count": feature_count,
                "numeric_feature_count": len(numeric_features),
                "non_numeric_feature_count": non_numeric_count,
                "missing_feature_count": missing_feature_count,
                "sample_size": sample_size,
                "sample_to_feature_ratio": sample_to_feature_ratio,
                "mean_missing_rate": mean_missing_rate,
                "high_missing_feature_count": high_missing_count,
                "constant_or_near_constant_feature_count": near_constant_count,
                "interpretation": _feature_inventory_interpretation(
                    sample_size,
                    feature_count,
                    sample_to_feature_ratio,
                    high_missing_count,
                    near_constant_count,
                ),
            }
        ],
        columns=FEATURE_INVENTORY_COLUMNS,
    )


def build_feature_audit_warnings(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    family_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return compact warnings for missingness, stability, concentration, and complexity."""

    if panel.empty:
        return _empty_feature_warnings()

    features = _coerce_feature_columns(panel, feature_columns)
    present_features = [column for column in features if column in panel]
    numeric_features = _numeric_audit_features(panel, present_features)
    numeric_data = (
        panel[numeric_features].apply(pd.to_numeric, errors="coerce")
        if numeric_features
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []

    missing_features = [column for column in features if column not in panel]
    if missing_features:
        rows.append(
            {
                "warning": "missing feature columns",
                "severity": "warning",
                "feature_count": len(missing_features),
                "detail": ", ".join(missing_features[:8]),
                "interpretation": "Configured feature columns were not present in this sample.",
            }
        )

    if not numeric_data.empty:
        missing_rates = numeric_data.isna().mean()
        high_missing = missing_rates[missing_rates >= HIGH_MISSING_RATE].sort_values(ascending=False)
        if not high_missing.empty:
            rows.append(
                {
                    "warning": "high missingness",
                    "severity": "warning",
                    "feature_count": int(len(high_missing)),
                    "detail": ", ".join(high_missing.index[:8]),
                    "interpretation": "Some features have high missingness and may weaken validation evidence.",
                }
            )

        near_constant = _near_constant_features(numeric_data)
        if near_constant:
            rows.append(
                {
                    "warning": "constant or near-constant features",
                    "severity": "warning",
                    "feature_count": len(near_constant),
                    "detail": ", ".join(near_constant[:8]),
                    "interpretation": "Some features show little variation in this sample.",
                }
            )

    feature_count = len(features)
    sample_to_feature_ratio = len(panel) / feature_count if feature_count else None
    if sample_to_feature_ratio is not None and sample_to_feature_ratio < LOW_SAMPLE_TO_FEATURE_RATIO:
        rows.append(
            {
                "warning": "low sample-to-feature ratio",
                "severity": "warning",
                "feature_count": feature_count,
                "detail": f"{sample_to_feature_ratio:.2f} samples per feature",
                "interpretation": "The feature set is wide relative to the sample, so overfit risk is higher.",
            }
        )

    families = family_summary if family_summary is not None else build_feature_family_summary(features)
    if not families.empty:
        dominant = families[families["share_of_features"] >= FAMILY_DOMINANCE_SHARE]
        if not dominant.empty:
            row = dominant.sort_values("share_of_features", ascending=False).iloc[0]
            rows.append(
                {
                    "warning": "feature family concentration",
                    "severity": "info",
                    "feature_count": int(row["feature_count"]),
                    "detail": str(row["family"]),
                    "interpretation": "One feature family represents most of the feature set.",
                }
            )

    complex_features = [
        feature for feature in features if _feature_family(feature) == "Fourier / wavelet / complex transform"
    ]
    complex_ratio = len(panel) / len(complex_features) if complex_features else None
    if complex_ratio is not None and complex_ratio < COMPLEX_FEATURE_SAMPLE_RATIO:
        rows.append(
            {
                "warning": "complex transform features with limited sample",
                "severity": "warning",
                "feature_count": len(complex_features),
                "detail": ", ".join(complex_features[:8]),
                "interpretation": "Complex transform features may be overfit-prone when sample depth is limited.",
            }
        )

    if not rows:
        return pd.DataFrame(
            [
                {
                    "warning": "no material feature audit warnings",
                    "severity": "info",
                    "feature_count": 0,
                    "detail": "",
                    "interpretation": "No material feature audit warnings were detected in this sample.",
                }
            ],
            columns=FEATURE_WARNING_COLUMNS,
        )
    return pd.DataFrame(rows, columns=FEATURE_WARNING_COLUMNS)


def build_feature_redundancy_summary(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    correlation_threshold: float = HIGH_CORRELATION_THRESHOLD,
    max_pairs: int = MAX_CORRELATION_PAIRS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return high-correlation feature-pair diagnostics without dumping a full matrix."""

    if panel.empty:
        return _empty_feature_redundancy_summary(), _empty_feature_correlation_pairs()

    features = _coerce_feature_columns(panel, feature_columns)
    numeric_features = _numeric_audit_features(panel, features)
    if len(numeric_features) < 2:
        summary = pd.DataFrame(
            [
                {
                    "numeric_feature_count": len(numeric_features),
                    "high_correlation_pair_count": 0,
                    "correlation_threshold": correlation_threshold,
                    "interpretation": "Not enough numeric features were available for redundancy diagnostics.",
                }
            ],
            columns=FEATURE_REDUNDANCY_SUMMARY_COLUMNS,
        )
        return summary, _empty_feature_correlation_pairs()

    data = panel[numeric_features].apply(pd.to_numeric, errors="coerce")
    corr = data.corr(numeric_only=True).abs()
    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(numeric_features):
        for right in numeric_features[left_index + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= correlation_threshold:
                pairs.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "abs_correlation": float(value),
                        "interpretation": "These features are highly correlated in this sample.",
                    }
                )
    pairs = sorted(pairs, key=lambda row: row["abs_correlation"], reverse=True)
    pair_count = len(pairs)
    interpretation = (
        "High-correlation feature pairs were detected; the feature set may contain redundant signals."
        if pair_count
        else "No high-correlation feature pairs were detected at this threshold."
    )
    summary = pd.DataFrame(
        [
            {
                "numeric_feature_count": len(numeric_features),
                "high_correlation_pair_count": pair_count,
                "correlation_threshold": correlation_threshold,
                "interpretation": interpretation,
            }
        ],
        columns=FEATURE_REDUNDANCY_SUMMARY_COLUMNS,
    )
    pair_frame = pd.DataFrame(pairs[:max_pairs], columns=FEATURE_CORRELATION_PAIR_COLUMNS)
    return summary, pair_frame


def _extract_model_step(model: object) -> object | None:
    if model is None:
        return None
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict) and "model" in named_steps:
        return named_steps["model"]
    return model


def build_feature_importance_summary(
    model: object | None,
    feature_columns: list[str],
    *,
    max_features: int = 20,
) -> pd.DataFrame:
    """Summarize feature importance from fitted models that expose simple attributes."""

    model_step = _extract_model_step(model)
    if model_step is None or not feature_columns:
        return _empty_feature_importance_summary()

    source = ""
    values: list[float] | None = None
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        values = [float(value) for value in importances]
        source = "feature_importances_"
    else:
        coefficients = getattr(model_step, "coef_", None)
        if coefficients is not None:
            frame = pd.DataFrame(coefficients)
            if not frame.empty:
                values = frame.abs().mean(axis=0).astype(float).tolist()
                source = "coef_"

    if values is None or len(values) != len(feature_columns):
        return _empty_feature_importance_summary()

    rows = [
        {
            "feature": feature,
            "importance": importance,
            "rank": rank,
            "source": source,
            "interpretation": "Higher values indicate stronger model reliance in the fitted estimator.",
        }
        for rank, (feature, importance) in enumerate(
            sorted(
                zip(feature_columns, values, strict=False),
                key=lambda item: item[1],
                reverse=True,
            ),
            start=1,
        )
    ]
    return pd.DataFrame(rows[:max_features], columns=FEATURE_IMPORTANCE_COLUMNS)


def build_ml_feature_audit(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    model: object | None = None,
) -> MLFeatureAudit:
    """Build read-only diagnostics for current ML feature columns."""

    features = _coerce_feature_columns(panel, feature_columns)
    inventory = build_feature_inventory_summary(panel, features)
    family_summary = build_feature_family_summary(features)
    warnings = build_feature_audit_warnings(panel, features, family_summary)
    redundancy_summary, high_correlation_pairs = build_feature_redundancy_summary(panel, features)
    present_numeric_features = _numeric_audit_features(panel, features)
    return MLFeatureAudit(
        inventory,
        family_summary,
        warnings,
        redundancy_summary,
        high_correlation_pairs,
        build_feature_importance_summary(model, present_numeric_features),
    )


def _return_target_column(panel: pd.DataFrame, horizon: int) -> str | None:
    for column in (f"forward_{horizon}d_excess_return", f"forward_{horizon}d_return"):
        if column in panel:
            return column
    return None


def _return_label_column(panel: pd.DataFrame, horizon: int) -> str | None:
    column = f"label_outperform_{horizon}d"
    return column if column in panel else None


def _drawdown_label_column(panel: pd.DataFrame, horizon: int) -> str | None:
    column = f"label_drawdown_risk_{horizon}d"
    return column if column in panel else None


def _spearman_relation(left: pd.Series, right: pd.Series) -> object:
    data = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if (
        len(data) < MIN_FEATURE_SIGNAL_SAMPLE_SIZE
        or data["left"].nunique(dropna=True) < 2
        or data["right"].nunique(dropna=True) < 2
    ):
        return pd.NA
    value = data["left"].corr(data["right"], method="spearman")
    return float(value) if pd.notna(value) else pd.NA


def _abs_or_na(value: object) -> object:
    return abs(float(value)) if pd.notna(value) else pd.NA


def _feature_signal_strength(row: pd.Series) -> object:
    values = pd.to_numeric(
        pd.Series(
            [
                row.get("abs_spearman_to_return_target"),
                row.get("abs_spearman_to_return_label"),
                row.get("abs_spearman_to_drawdown_label"),
            ]
        ),
        errors="coerce",
    ).dropna()
    return float(values.max()) if not values.empty else pd.NA


def _quantile_signal_stats(feature: pd.Series, target: pd.Series) -> dict[str, object]:
    data = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature, errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    ).dropna()
    if (
        len(data) < MIN_FEATURE_SIGNAL_SAMPLE_SIZE
        or data["feature"].nunique(dropna=True) < MIN_FEATURE_SIGNAL_UNIQUE_VALUES
        or data["target"].nunique(dropna=True) < 2
    ):
        return {
            "valid_sample_count": int(len(data)),
            "bottom_quantile_target_mean": pd.NA,
            "top_quantile_target_mean": pd.NA,
            "quantile_spread": pd.NA,
            "abs_quantile_spread": pd.NA,
        }

    bucket_count = min(FEATURE_SIGNAL_QUANTILE_COUNT, int(data["feature"].nunique(dropna=True)))
    try:
        buckets = pd.qcut(data["feature"], q=bucket_count, duplicates="drop")
    except ValueError:
        return {
            "valid_sample_count": int(len(data)),
            "bottom_quantile_target_mean": pd.NA,
            "top_quantile_target_mean": pd.NA,
            "quantile_spread": pd.NA,
            "abs_quantile_spread": pd.NA,
        }
    if buckets.nunique(dropna=True) < 2:
        return {
            "valid_sample_count": int(len(data)),
            "bottom_quantile_target_mean": pd.NA,
            "top_quantile_target_mean": pd.NA,
            "quantile_spread": pd.NA,
            "abs_quantile_spread": pd.NA,
        }

    means = data.assign(_bucket=buckets).groupby("_bucket", observed=True)["target"].mean()
    bottom_mean = float(means.iloc[0])
    top_mean = float(means.iloc[-1])
    spread = top_mean - bottom_mean
    return {
        "valid_sample_count": int(len(data)),
        "bottom_quantile_target_mean": bottom_mean,
        "top_quantile_target_mean": top_mean,
        "quantile_spread": float(spread),
        "abs_quantile_spread": abs(float(spread)),
    }


def _feature_signal_interpretation(
    valid_sample_count: int,
    unique_value_count: int,
    signal_strength: object,
) -> str:
    if valid_sample_count < MIN_FEATURE_SIGNAL_SAMPLE_SIZE:
        return "Too few valid observations for univariate signal diagnostics."
    if unique_value_count < MIN_FEATURE_SIGNAL_UNIQUE_VALUES:
        return "Feature has too few distinct values for stable univariate diagnostics."
    if pd.isna(signal_strength):
        return "Target or label relation was unavailable for this feature."
    if float(signal_strength) < NEAR_ZERO_FEATURE_SIGNAL:
        return "Univariate relation is near zero in this sample."
    if float(signal_strength) >= 0.20:
        return "Feature has stronger univariate relation in this sample."
    return "Feature shows a modest univariate relation in this sample."


def build_feature_signal_table(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    horizon: int = 20,
    max_features: int | None = FEATURE_SIGNAL_TOP_N,
) -> pd.DataFrame:
    """Summarize simple univariate feature relation to existing targets and labels."""

    if panel.empty:
        return _empty_feature_signal_table()

    features = _coerce_feature_columns(panel, feature_columns)
    numeric_features = _numeric_audit_features(panel, features)
    if not numeric_features:
        return _empty_feature_signal_table()

    return_target = _return_target_column(panel, horizon)
    return_label = _return_label_column(panel, horizon)
    drawdown_label = _drawdown_label_column(panel, horizon)
    rows: list[dict[str, object]] = []
    for feature in numeric_features:
        feature_values = pd.to_numeric(panel[feature], errors="coerce")
        valid_sample_count = int(feature_values.notna().sum())
        unique_value_count = int(feature_values.nunique(dropna=True))
        missing_rate = float(feature_values.isna().mean())

        spearman_to_target = (
            _spearman_relation(feature_values, panel[return_target]) if return_target is not None else pd.NA
        )
        abs_spearman_to_return_label = (
            _abs_or_na(_spearman_relation(feature_values, panel[return_label]))
            if return_label is not None
            else pd.NA
        )
        abs_spearman_to_drawdown_label = (
            _abs_or_na(_spearman_relation(feature_values, panel[drawdown_label]))
            if drawdown_label is not None
            else pd.NA
        )
        quantile_stats = (
            _quantile_signal_stats(feature_values, panel[return_target])
            if return_target is not None
            else {
                "bottom_quantile_target_mean": pd.NA,
                "top_quantile_target_mean": pd.NA,
                "quantile_spread": pd.NA,
            }
        )
        row = {
            "feature": feature,
            "family": _feature_family(feature),
            "valid_sample_count": valid_sample_count,
            "missing_rate": missing_rate,
            "unique_value_count": unique_value_count,
            "abs_spearman_to_return_target": _abs_or_na(spearman_to_target),
            "spearman_to_return_target": spearman_to_target,
            "abs_spearman_to_return_label": abs_spearman_to_return_label,
            "abs_spearman_to_drawdown_label": abs_spearman_to_drawdown_label,
            "top_quantile_target_mean": quantile_stats["top_quantile_target_mean"],
            "bottom_quantile_target_mean": quantile_stats["bottom_quantile_target_mean"],
            "quantile_spread": quantile_stats["quantile_spread"],
        }
        row["interpretation"] = _feature_signal_interpretation(
            valid_sample_count,
            unique_value_count,
            _feature_signal_strength(pd.Series(row)),
        )
        rows.append(row)

    if not rows:
        return _empty_feature_signal_table()
    signal_table = pd.DataFrame(rows, columns=FEATURE_SIGNAL_COLUMNS)
    strengths = signal_table.apply(_feature_signal_strength, axis=1)
    sorted_table = (
        signal_table.assign(_signal_strength=pd.to_numeric(strengths, errors="coerce"))
        .sort_values(["_signal_strength", "feature"], ascending=[False, True], na_position="last")
        .drop(columns="_signal_strength")
        .reset_index(drop=True)
    )
    return sorted_table if max_features is None else sorted_table.head(max_features)


def build_feature_family_signal_summary(
    signal_table: pd.DataFrame,
    *,
    top_feature_count: int = FEATURE_SIGNAL_TOP_N,
) -> pd.DataFrame:
    """Group feature signal diagnostics by existing feature-name families."""

    if signal_table.empty:
        return _empty_feature_family_signal_summary()

    data = signal_table.copy()
    data["_signal_strength"] = pd.to_numeric(data.apply(_feature_signal_strength, axis=1), errors="coerce")
    ranked = data.dropna(subset=["_signal_strength"]).sort_values("_signal_strength", ascending=False)
    top_features = set(ranked.head(top_feature_count)["feature"])
    rows: list[dict[str, object]] = []
    for family, group in data.groupby("family", sort=False):
        strengths = group["_signal_strength"].dropna()
        if strengths.empty:
            top_feature = ""
            top_signal = pd.NA
            median_signal = pd.NA
            max_signal = pd.NA
            interpretation = "Feature-family signal was unavailable in this sample."
        else:
            top_row = group.sort_values("_signal_strength", ascending=False).iloc[0]
            top_feature = str(top_row["feature"])
            top_signal = float(top_row["_signal_strength"])
            median_signal = float(strengths.median())
            max_signal = float(strengths.max())
            interpretation = (
                "This family contains one of the stronger univariate signals."
                if top_signal >= 0.20
                else "This family shows modest or weak univariate signal in this sample."
            )
        rows.append(
            {
                "family": family,
                "feature_count": int(len(group)),
                "median_abs_signal": median_signal,
                "max_abs_signal": max_signal,
                "top_feature": top_feature,
                "top_feature_signal": top_signal,
                "share_of_top_features": (
                    float(group["feature"].isin(top_features).sum() / len(top_features))
                    if top_features
                    else pd.NA
                ),
                "interpretation": interpretation,
            }
        )
    return (
        pd.DataFrame(rows, columns=FEATURE_FAMILY_SIGNAL_COLUMNS)
        .sort_values(["max_abs_signal", "family"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )


def build_feature_quantile_signal_summary(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    horizon: int = 20,
    max_features: int = 10,
) -> pd.DataFrame:
    """Return top feature quantile target-spread diagnostics where a numeric target exists."""

    if panel.empty:
        return _empty_feature_quantile_signal_summary()

    return_target = _return_target_column(panel, horizon)
    if return_target is None:
        return _empty_feature_quantile_signal_summary()

    features = _coerce_feature_columns(panel, feature_columns)
    numeric_features = _numeric_audit_features(panel, features)
    rows: list[dict[str, object]] = []
    for feature in numeric_features:
        stats = _quantile_signal_stats(panel[feature], panel[return_target])
        if pd.isna(stats["quantile_spread"]):
            continue
        rows.append(
            {
                "feature": feature,
                "family": _feature_family(feature),
                "valid_sample_count": stats["valid_sample_count"],
                "bottom_quantile_target_mean": stats["bottom_quantile_target_mean"],
                "top_quantile_target_mean": stats["top_quantile_target_mean"],
                "quantile_spread": stats["quantile_spread"],
                "abs_quantile_spread": stats["abs_quantile_spread"],
                "interpretation": "High-minus-low feature quantiles show numeric target separation.",
            }
        )
    if not rows:
        return _empty_feature_quantile_signal_summary()
    return (
        pd.DataFrame(rows, columns=FEATURE_QUANTILE_SIGNAL_COLUMNS)
        .sort_values(["abs_quantile_spread", "feature"], ascending=[False, True])
        .head(max_features)
        .reset_index(drop=True)
    )


def build_feature_signal_warnings(
    panel: pd.DataFrame,
    signal_table: pd.DataFrame,
    family_summary: pd.DataFrame,
    high_correlation_pairs: pd.DataFrame | None = None,
    *,
    horizon: int = 20,
) -> pd.DataFrame:
    """Return compact caution rows for univariate feature-signal diagnostics."""

    rows: list[dict[str, object]] = []
    if panel.empty or signal_table.empty:
        rows.append(
            {
                "warning": "feature signal unavailable",
                "severity": "warning",
                "feature_count": 0,
                "detail": "empty sample or no numeric features",
                "interpretation": "No suitable feature-signal diagnostics were available.",
            }
        )
        return pd.DataFrame(rows, columns=FEATURE_WARNING_COLUMNS)

    if (
        _return_target_column(panel, horizon) is None
        and _return_label_column(panel, horizon) is None
        and _drawdown_label_column(panel, horizon) is None
    ):
        rows.append(
            {
                "warning": "no suitable target or label",
                "severity": "warning",
                "feature_count": int(len(signal_table)),
                "detail": f"horizon={horizon}",
                "interpretation": "No existing return target or supervised label was available.",
            }
        )

    too_few_samples = signal_table[signal_table["valid_sample_count"] < MIN_FEATURE_SIGNAL_SAMPLE_SIZE]
    if not too_few_samples.empty:
        rows.append(
            {
                "warning": "too few valid samples",
                "severity": "warning",
                "feature_count": int(len(too_few_samples)),
                "detail": ", ".join(too_few_samples["feature"].head(8)),
                "interpretation": "Some features lack enough valid observations for stable signal diagnostics.",
            }
        )

    too_few_values = signal_table[signal_table["unique_value_count"] < MIN_FEATURE_SIGNAL_UNIQUE_VALUES]
    if not too_few_values.empty:
        rows.append(
            {
                "warning": "too few unique values",
                "severity": "warning",
                "feature_count": int(len(too_few_values)),
                "detail": ", ".join(too_few_values["feature"].head(8)),
                "interpretation": "Some features have too few distinct values for stable signal diagnostics.",
            }
        )

    strengths = pd.to_numeric(signal_table.apply(_feature_signal_strength, axis=1), errors="coerce").dropna()
    if strengths.empty:
        rows.append(
            {
                "warning": "target relation unavailable",
                "severity": "warning",
                "feature_count": int(len(signal_table)),
                "detail": "",
                "interpretation": "Target or label relation was unavailable for the current feature set.",
            }
        )
    elif float((strengths < NEAR_ZERO_FEATURE_SIGNAL).mean()) >= 0.75:
        rows.append(
            {
                "warning": "near-zero univariate signal",
                "severity": "info",
                "feature_count": int((strengths < NEAR_ZERO_FEATURE_SIGNAL).sum()),
                "detail": f"threshold={NEAR_ZERO_FEATURE_SIGNAL:.2f}",
                "interpretation": "Most features show near-zero univariate signal in this sample.",
            }
        )

    if not family_summary.empty and "share_of_top_features" in family_summary:
        complex_rows = family_summary[
            family_summary["family"].eq("Fourier / wavelet / complex transform")
            & (pd.to_numeric(family_summary["share_of_top_features"], errors="coerce") >= COMPLEX_TOP_SIGNAL_SHARE)
        ]
        if not complex_rows.empty:
            rows.append(
                {
                    "warning": "complex feature signal concentration",
                    "severity": "info",
                    "feature_count": int(complex_rows["feature_count"].iloc[0]),
                    "detail": "Fourier / wavelet / complex transform",
                    "interpretation": "Top univariate signal is concentrated in complex transform features.",
                }
            )

    if high_correlation_pairs is not None and not high_correlation_pairs.empty:
        top_signal_features = set(signal_table.head(5)["feature"])
        redundant_features = set(high_correlation_pairs["feature_1"]).union(set(high_correlation_pairs["feature_2"]))
        overlap = sorted(top_signal_features & redundant_features)
        if overlap:
            rows.append(
                {
                    "warning": "top signal features are redundant",
                    "severity": "info",
                    "feature_count": len(overlap),
                    "detail": ", ".join(overlap[:8]),
                    "interpretation": "Some top-ranked univariate features are also highly correlated with peers.",
                }
            )

    if not rows:
        rows.append(
            {
                "warning": "no material feature signal warnings",
                "severity": "info",
                "feature_count": 0,
                "detail": "",
                "interpretation": "No material feature signal warnings were detected in this sample.",
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_WARNING_COLUMNS)


def build_ml_feature_signal_diagnostics(
    panel: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    horizon: int = 20,
    high_correlation_pairs: pd.DataFrame | None = None,
) -> MLFeatureSignalDiagnostics:
    """Build read-only univariate signal diagnostics for current ML features."""

    full_signal_table = build_feature_signal_table(panel, feature_columns, horizon=horizon, max_features=None)
    signal_table = full_signal_table.head(FEATURE_SIGNAL_TOP_N)
    family_summary = build_feature_family_signal_summary(full_signal_table)
    quantile_summary = build_feature_quantile_signal_summary(panel, feature_columns, horizon=horizon)
    warnings = build_feature_signal_warnings(
        panel,
        full_signal_table,
        family_summary,
        high_correlation_pairs,
        horizon=horizon,
    )
    return MLFeatureSignalDiagnostics(signal_table, family_summary, quantile_summary, warnings)


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
    forward_return_column: str = "forward_excess_return",
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
    forward_return_col: str = "forward_excess_return",
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


def _numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(data[column], errors="coerce")


def _spearman(data: pd.DataFrame, left: str, right: str) -> object:
    if left not in data or right not in data:
        return pd.NA
    usable = data[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(usable) < 2 or usable[left].nunique() < 2 or usable[right].nunique() < 2:
        return pd.NA
    value = usable[left].corr(usable[right], method="spearman")
    return pd.NA if pd.isna(value) else float(value)


def _score_direction(spread: object, sample_size: int, min_samples: int) -> str:
    if sample_size < min_samples or pd.isna(spread):
        return "insufficient"
    if spread > SPREAD_SIMILARITY_TOLERANCE:
        return "aligned"
    if spread < -SPREAD_SIMILARITY_TOLERANCE:
        return "inverted"
    return "flat"


def _score_direction_interpretation(direction: str) -> str:
    if direction == "aligned":
        return "Higher ML scores correspond to stronger realised forward excess returns in this sample."
    if direction == "inverted":
        return "Higher ML scores correspond to weaker realised forward excess returns in this sample."
    if direction == "flat":
        return "Forward excess returns are similar across high and low ML score buckets in this sample."
    return "The usable sample is too small or incomplete for a score-direction read."


def _bucket_score_data(
    data: pd.DataFrame,
    score_col: str,
    *,
    bucket_col: str = "score_bucket",
) -> pd.DataFrame:
    bucketed = data.copy()
    bucketed[bucket_col] = pd.cut(
        bucketed[score_col],
        bins=[-float("inf"), 40.0, 70.0, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    return bucketed


def _bucket_spread(
    data: pd.DataFrame,
    score_col: str,
    target_col: str,
    min_bucket_size: int,
) -> tuple[int, object, object, object]:
    if score_col not in data or target_col not in data:
        return 0, pd.NA, pd.NA, pd.NA

    usable = data[[score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if usable.empty:
        return 0, pd.NA, pd.NA, pd.NA

    bucketed = _bucket_score_data(usable, score_col)
    bottom = bucketed[bucketed["score_bucket"] == "Low"]
    top = bucketed[bucketed["score_bucket"] == "High"]
    if len(bottom) < min_bucket_size or len(top) < min_bucket_size:
        return len(usable), pd.NA, pd.NA, pd.NA

    top_return = float(top[target_col].mean())
    bottom_return = float(bottom[target_col].mean())
    return len(usable), top_return, bottom_return, top_return - bottom_return


def build_ml_score_direction_diagnostics(
    score_panel: pd.DataFrame,
    *,
    score_col: str = "ML Score",
    target_col: str = "forward_excess_return",
    label_col: str = "actual_out",
    drawdown_label_col: str = "actual_risk",
    min_bucket_size: int = MIN_SCORE_DIRECTION_BUCKET_COUNT,
    min_samples: int = MIN_SCORE_DIRECTION_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Summarize whether the existing ML score direction matches realised outcomes."""

    if score_panel.empty or score_col not in score_panel:
        return _empty_score_direction_summary()

    sample_size, top_return, bottom_return, spread = _bucket_spread(
        score_panel,
        score_col,
        target_col,
        min_bucket_size,
    )
    direction = _score_direction(spread, sample_size, min_samples)
    return pd.DataFrame(
        [
            {
                "sample_size": sample_size,
                "score_column": score_col if score_col in score_panel else pd.NA,
                "target_column": target_col if target_col in score_panel else pd.NA,
                "label_column": label_col if label_col in score_panel else pd.NA,
                "drawdown_label_column": drawdown_label_col if drawdown_label_col in score_panel else pd.NA,
                "score_to_forward_return_spearman": _spearman(score_panel, score_col, target_col),
                "score_to_return_label_spearman": _spearman(score_panel, score_col, label_col),
                "score_to_drawdown_label_spearman": _spearman(
                    score_panel,
                    score_col,
                    drawdown_label_col,
                ),
                "top_bucket_forward_return": top_return,
                "bottom_bucket_forward_return": bottom_return,
                "top_minus_bottom_spread": spread,
                "direction": direction,
                "interpretation": _score_direction_interpretation(direction),
            }
        ],
        columns=SCORE_DIRECTION_SUMMARY_COLUMNS,
    )


def _alignment_row(
    data: pd.DataFrame,
    *,
    diagnostic: str,
    score_col: str,
    value_col: str,
    positive_text: str,
    negative_text: str,
    min_bucket_size: int,
) -> dict[str, object] | None:
    if score_col not in data or value_col not in data:
        return None

    usable = data[[score_col, value_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if usable.empty:
        return None

    bucketed = _bucket_score_data(usable, score_col)
    bottom = bucketed[bucketed["score_bucket"] == "Low"]
    top = bucketed[bucketed["score_bucket"] == "High"]
    if len(bottom) < min_bucket_size or len(top) < min_bucket_size:
        return {
            "diagnostic": diagnostic,
            "sample_size": int(len(usable)),
            "bottom_score_bucket_mean": pd.NA,
            "top_score_bucket_mean": pd.NA,
            "top_minus_bottom_spread": pd.NA,
            "higher_score_corresponds_to": "insufficient",
            "interpretation": "The score buckets are too small for this alignment check.",
        }

    bottom_mean = float(bottom[value_col].mean())
    top_mean = float(top[value_col].mean())
    spread = top_mean - bottom_mean
    if spread > SPREAD_SIMILARITY_TOLERANCE:
        corresponds_to = positive_text
    elif spread < -SPREAD_SIMILARITY_TOLERANCE:
        corresponds_to = negative_text
    else:
        corresponds_to = "similar values"

    return {
        "diagnostic": diagnostic,
        "sample_size": int(len(usable)),
        "bottom_score_bucket_mean": bottom_mean,
        "top_score_bucket_mean": top_mean,
        "top_minus_bottom_spread": spread,
        "higher_score_corresponds_to": corresponds_to,
        "interpretation": f"Higher score buckets show {corresponds_to} in this sample.",
    }


def build_probability_label_alignment(
    score_panel: pd.DataFrame,
    *,
    score_col: str = "ML Score",
    return_probability_col: str = "probability_out",
    target_col: str = "forward_excess_return",
    return_label_col: str = "actual_out",
    drawdown_label_col: str = "actual_risk",
    forward_drawdown_col: str = "forward_drawdown",
    min_bucket_size: int = MIN_SCORE_DIRECTION_BUCKET_COUNT,
) -> pd.DataFrame:
    """Show how higher existing ML scores line up with probabilities, labels, and realised outcomes."""

    if score_panel.empty or score_col not in score_panel:
        return _empty_probability_label_alignment()

    rows = [
        _alignment_row(
            score_panel,
            diagnostic="outperformance probability",
            score_col=score_col,
            value_col=return_probability_col,
            positive_text="higher outperformance probability",
            negative_text="lower outperformance probability",
            min_bucket_size=min_bucket_size,
        ),
        _alignment_row(
            score_panel,
            diagnostic="realised forward excess return",
            score_col=score_col,
            value_col=target_col,
            positive_text="higher realised forward excess return",
            negative_text="lower realised forward excess return",
            min_bucket_size=min_bucket_size,
        ),
        _alignment_row(
            score_panel,
            diagnostic="outperformance label rate",
            score_col=score_col,
            value_col=return_label_col,
            positive_text="higher outperformance label rate",
            negative_text="lower outperformance label rate",
            min_bucket_size=min_bucket_size,
        ),
        _alignment_row(
            score_panel,
            diagnostic="drawdown-risk label rate",
            score_col=score_col,
            value_col=drawdown_label_col,
            positive_text="higher drawdown-risk label rate",
            negative_text="lower drawdown-risk label rate",
            min_bucket_size=min_bucket_size,
        ),
        _alignment_row(
            score_panel,
            diagnostic="realised drawdown event rate",
            score_col=score_col,
            value_col=drawdown_label_col,
            positive_text="higher realised drawdown event rate",
            negative_text="lower realised drawdown event rate",
            min_bucket_size=min_bucket_size,
        ),
        _alignment_row(
            score_panel,
            diagnostic="realised forward drawdown",
            score_col=score_col,
            value_col=forward_drawdown_col,
            positive_text="less severe realised drawdowns",
            negative_text="more severe realised drawdowns",
            min_bucket_size=min_bucket_size,
        ),
    ]
    present_rows = [row for row in rows if row is not None]
    if not present_rows:
        return _empty_probability_label_alignment()
    return pd.DataFrame(present_rows, columns=PROBABILITY_LABEL_ALIGNMENT_COLUMNS)


def _monotonicity_result(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient"
    diffs = pd.Series(values).diff().dropna()
    if bool((diffs > SPREAD_SIMILARITY_TOLERANCE).all()):
        return "aligned"
    if bool((diffs < -SPREAD_SIMILARITY_TOLERANCE).all()):
        return "inverted"
    if bool((diffs.abs() <= SPREAD_SIMILARITY_TOLERANCE).all()):
        return "flat"
    return "mixed"


def _monotonicity_interpretation(result: str) -> str:
    if result == "aligned":
        return "Forward excess returns rise across the ML score buckets in this sample."
    if result == "inverted":
        return "Forward excess returns fall across the ML score buckets in this sample."
    if result == "flat":
        return "Forward excess returns are similar across ML score buckets in this sample."
    if result == "mixed":
        return "Forward excess returns are not monotonic across ML score buckets in this sample."
    return "There is too little usable bucket data for a monotonicity read."


def build_score_bucket_monotonicity(
    score_panel: pd.DataFrame,
    *,
    score_col: str = "ML Score",
    target_col: str = "forward_excess_return",
    return_label_col: str = "actual_out",
    drawdown_label_col: str = "actual_risk",
) -> pd.DataFrame:
    """Return per-bucket outcome rates and a compact monotonicity read."""

    if score_panel.empty or score_col not in score_panel:
        return _empty_score_bucket_monotonicity()

    data = score_panel.copy()
    required = [score_col, target_col]
    if any(column not in data for column in required):
        return _empty_score_bucket_monotonicity()
    data[score_col] = _numeric_column(data, score_col)
    data[target_col] = _numeric_column(data, target_col)
    data = data.dropna(subset=[score_col, target_col])
    if data.empty:
        return _empty_score_bucket_monotonicity()

    data = _bucket_score_data(data, score_col)
    grouped = data.groupby("score_bucket", observed=True)
    rows: list[dict[str, object]] = []
    bucket_returns: list[float] = []
    for bucket in ("Low", "Medium", "High"):
        if bucket not in grouped.groups:
            continue
        group = grouped.get_group(bucket)
        mean_return = float(group[target_col].mean())
        bucket_returns.append(mean_return)
        rows.append(
            {
                "bucket": bucket,
                "sample_size": int(len(group)),
                "mean_score": float(group[score_col].mean()),
                "mean_forward_return": mean_return,
                "return_label_rate": float(_numeric_column(group, return_label_col).mean())
                if return_label_col in group
                else pd.NA,
                "drawdown_label_rate": float(_numeric_column(group, drawdown_label_col).mean())
                if drawdown_label_col in group
                else pd.NA,
                "monotonicity_result": "",
                "interpretation": "",
            }
        )

    result = _monotonicity_result(bucket_returns)
    interpretation = _monotonicity_interpretation(result)
    for row in rows:
        row["monotonicity_result"] = result
        row["interpretation"] = interpretation

    return pd.DataFrame(rows, columns=SCORE_BUCKET_MONOTONICITY_COLUMNS)


def _rank_quantile_bucket(values: pd.Series, labels: list[str]) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def _joint_cell_interpretation(opportunity_bucket: str, risk_bucket: str) -> str:
    if opportunity_bucket == "High opportunity" and risk_bucket == "Low risk":
        return "Best setup candidate if returns lead and drawdown stays controlled."
    if opportunity_bucket == "High opportunity" and risk_bucket == "High risk":
        return "High-opportunity setup with elevated pullback risk; compare drawdown against high opportunity / low risk."
    if opportunity_bucket == "Low opportunity" and risk_bucket == "Low risk":
        return "Defensive or low-conviction setup with lower opportunity and lower pullback risk."
    return "Worst setup candidate if returns lag and drawdown risk is elevated."


def _joint_validation_result(
    cells: pd.DataFrame,
    *,
    min_cell_size: int,
) -> str:
    lookup = cells.set_index(["opportunity_bucket", "risk_bucket"])
    required_cells = [
        ("High opportunity", "Low risk"),
        ("High opportunity", "High risk"),
        ("Low opportunity", "Low risk"),
        ("Low opportunity", "High risk"),
    ]
    if any(lookup.loc[cell, "sample_size"] < min_cell_size for cell in required_cells):
        return "insufficient data to compare"

    high_low = lookup.loc[("High opportunity", "Low risk")]
    high_high = lookup.loc[("High opportunity", "High risk")]
    low_high = lookup.loc[("Low opportunity", "High risk")]
    high_low_return = high_low["avg_forward_excess_return"]
    low_high_return = low_high["avg_forward_excess_return"]
    if pd.isna(high_low_return) or pd.isna(low_high_return):
        return "insufficient data to compare"

    return_support = high_low_return - low_high_return > SPREAD_SIMILARITY_TOLERANCE
    high_low_drawdown = high_low["avg_forward_drawdown"]
    high_high_drawdown = high_high["avg_forward_drawdown"]
    high_low_event_rate = high_low["drawdown_event_rate"]
    high_high_event_rate = high_high["drawdown_event_rate"]
    drawdown_support = False
    if pd.notna(high_low_drawdown) and pd.notna(high_high_drawdown):
        drawdown_support = high_high_drawdown < high_low_drawdown - SPREAD_SIMILARITY_TOLERANCE
    if pd.notna(high_low_event_rate) and pd.notna(high_high_event_rate):
        drawdown_support = (
            drawdown_support
            or high_high_event_rate - high_low_event_rate > SPREAD_SIMILARITY_TOLERANCE
        )

    if return_support and drawdown_support:
        return "joint validation supports separate opportunity and risk signals"
    if not return_support and not drawdown_support:
        return "joint validation does not support the combined signal"
    return "joint validation is mixed"


def build_opportunity_risk_joint_validation(
    score_panel: pd.DataFrame,
    *,
    opportunity_col: str = "probability_out",
    risk_col: str = "probability_risk",
    return_col: str = "forward_excess_return",
    drawdown_col: str = "forward_drawdown",
    drawdown_label_col: str = "actual_risk",
    min_cell_size: int = MIN_JOINT_VALIDATION_CELL_COUNT,
    min_samples: int = MIN_JOINT_VALIDATION_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Compare existing opportunity and pullback-risk validation signals in a 2x2 matrix."""

    if score_panel.empty:
        return _empty_opportunity_risk_joint_validation()
    required = [opportunity_col, risk_col, return_col]
    if any(column not in score_panel for column in required):
        return _empty_opportunity_risk_joint_validation()

    data = score_panel.copy()
    for column in required:
        data[column] = _numeric_column(data, column)
    optional_columns = [column for column in (drawdown_col, drawdown_label_col) if column in data]
    for column in optional_columns:
        data[column] = _numeric_column(data, column)
    data = data.dropna(subset=required)
    if (
        len(data) < min_samples
        or data[opportunity_col].nunique(dropna=True) < 2
        or data[risk_col].nunique(dropna=True) < 2
    ):
        return _empty_opportunity_risk_joint_validation()

    data["opportunity_bucket"] = _rank_quantile_bucket(
        data[opportunity_col],
        ["Low opportunity", "High opportunity"],
    )
    data["risk_bucket"] = _rank_quantile_bucket(data[risk_col], ["Low risk", "High risk"])

    rows: list[dict[str, object]] = []
    for opportunity_bucket in ("High opportunity", "Low opportunity"):
        for risk_bucket in ("Low risk", "High risk"):
            cell = data[
                (data["opportunity_bucket"] == opportunity_bucket)
                & (data["risk_bucket"] == risk_bucket)
            ]
            rows.append(
                {
                    "opportunity_bucket": opportunity_bucket,
                    "risk_bucket": risk_bucket,
                    "sample_size": int(len(cell)),
                    "avg_forward_excess_return": float(cell[return_col].mean()) if not cell.empty else pd.NA,
                    "avg_forward_drawdown": (
                        float(cell[drawdown_col].mean())
                        if drawdown_col in cell and not cell[drawdown_col].dropna().empty
                        else pd.NA
                    ),
                    "drawdown_event_rate": (
                        float(cell[drawdown_label_col].mean())
                        if drawdown_label_col in cell and not cell[drawdown_label_col].dropna().empty
                        else pd.NA
                    ),
                    "interpretation": _joint_cell_interpretation(opportunity_bucket, risk_bucket),
                    "joint_validation_result": "",
                }
            )

    result = pd.DataFrame(rows, columns=OPPORTUNITY_RISK_JOINT_COLUMNS)
    joint_result = _joint_validation_result(result, min_cell_size=min_cell_size)
    result["joint_validation_result"] = joint_result
    return result


def interpret_opportunity_risk_joint_validation(joint_validation: pd.DataFrame) -> str:
    """Return the compact Research Lab readout for the joint validation table."""

    if joint_validation.empty or "joint_validation_result" not in joint_validation:
        return "insufficient data to compare"
    result = joint_validation["joint_validation_result"].dropna()
    if result.empty:
        return "insufficient data to compare"
    return str(result.iloc[0])


def _target_comparison_row(
    predictions: pd.DataFrame,
    *,
    target_version: str,
    target_col: str,
    min_bucket_size: int,
) -> dict[str, object]:
    if predictions.empty or "probability" not in predictions or target_col not in predictions:
        return {
            "target_version": target_version,
            "target_column": target_col if target_col in predictions else pd.NA,
            "sample_size": 0,
            "low_bucket_target": pd.NA,
            "middle_bucket_target": pd.NA,
            "high_bucket_target": pd.NA,
            "high_minus_low_spread": pd.NA,
            "monotonicity": "insufficient",
            "baseline_spread": pd.NA,
            "relative_result": "insufficient data",
            "interpretation": "There is too little usable data for this target comparison.",
        }

    data = predictions[["probability", target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < min_bucket_size * 2 or data["probability"].nunique(dropna=True) < 2:
        return {
            "target_version": target_version,
            "target_column": target_col,
            "sample_size": int(len(data)),
            "low_bucket_target": pd.NA,
            "middle_bucket_target": pd.NA,
            "high_bucket_target": pd.NA,
            "high_minus_low_spread": pd.NA,
            "monotonicity": "insufficient",
            "baseline_spread": pd.NA,
            "relative_result": "insufficient data",
            "interpretation": "There is too little usable data for this target comparison.",
        }

    try:
        data = data.copy()
        data["_probability_bucket"] = pd.qcut(data["probability"], q=3, labels=False, duplicates="drop")
    except ValueError:
        data = data.copy()
        data["_probability_bucket"] = pd.NA

    data = data.dropna(subset=["_probability_bucket"])
    if data["_probability_bucket"].nunique(dropna=True) < 2:
        return {
            "target_version": target_version,
            "target_column": target_col,
            "sample_size": int(len(data)),
            "low_bucket_target": pd.NA,
            "middle_bucket_target": pd.NA,
            "high_bucket_target": pd.NA,
            "high_minus_low_spread": pd.NA,
            "monotonicity": "insufficient",
            "baseline_spread": pd.NA,
            "relative_result": "insufficient data",
            "interpretation": "Probability buckets were too narrow for this target comparison.",
        }

    grouped = data.groupby("_probability_bucket", observed=True)[target_col]
    counts = grouped.size()
    means = grouped.mean().sort_index()
    low_code = means.index.min()
    high_code = means.index.max()
    low_count = int(counts.loc[low_code])
    high_count = int(counts.loc[high_code])
    if low_count < min_bucket_size or high_count < min_bucket_size:
        spread = pd.NA
        monotonicity = "insufficient"
        interpretation = "Probability buckets were too small for this target comparison."
    else:
        spread = float(means.loc[high_code] - means.loc[low_code])
        monotonicity = _monotonicity_result([float(value) for value in means])
        interpretation = "Probability buckets show target separation for this diagnostics-only target."

    middle_values = means.iloc[1:-1]
    return {
        "target_version": target_version,
        "target_column": target_col,
        "sample_size": int(len(data)),
        "low_bucket_target": float(means.loc[low_code]) if low_count >= min_bucket_size else pd.NA,
        "middle_bucket_target": float(middle_values.mean()) if len(middle_values) else pd.NA,
        "high_bucket_target": float(means.loc[high_code]) if high_count >= min_bucket_size else pd.NA,
        "high_minus_low_spread": spread,
        "monotonicity": monotonicity,
        "baseline_spread": pd.NA,
        "relative_result": "",
        "interpretation": interpretation,
    }


def _target_comparison_score(row: pd.Series) -> tuple[float, int] | None:
    spread = pd.to_numeric(pd.Series([row.get("high_minus_low_spread")]), errors="coerce").iloc[0]
    if pd.isna(spread):
        return None
    monotonicity_rank = {
        "insufficient": 0,
        "inverted": 1,
        "flat": 2,
        "mixed": 2,
        "aligned": 3,
    }.get(str(row.get("monotonicity")), 0)
    return float(spread), monotonicity_rank


def _target_comparison_result(reference: pd.Series, candidate: pd.Series, version: str) -> str:
    v1_score = _target_comparison_score(reference)
    candidate_score = _target_comparison_score(candidate)
    if v1_score is None or candidate_score is None:
        return "insufficient data"

    v1_spread, v1_monotonicity = v1_score
    candidate_spread, candidate_monotonicity = candidate_score
    if (
        (
            candidate_spread > v1_spread + SPREAD_SIMILARITY_TOLERANCE
            and candidate_monotonicity >= v1_monotonicity
        )
        or (
            candidate_spread + SPREAD_SIMILARITY_TOLERANCE >= v1_spread
            and candidate_monotonicity > v1_monotonicity
        )
    ):
        return f"{version} looks better"
    if (
        candidate_spread + SPREAD_SIMILARITY_TOLERANCE < v1_spread
        or candidate_monotonicity < v1_monotonicity
    ):
        return f"{version} looks worse"
    return f"{version} looks similar"


def build_ml_target_comparison(
    outperformance_predictions: pd.DataFrame,
    risk_adjusted_predictions: pd.DataFrame,
    tail_risk_predictions: pd.DataFrame | None = None,
    *,
    min_bucket_size: int = MIN_SCORE_DIRECTION_BUCKET_COUNT,
) -> pd.DataFrame:
    """Compare v1 outperformance, v2 risk-adjusted, and v3 tail-risk targets."""

    rows = [
        _target_comparison_row(
            outperformance_predictions,
            target_version="v1 outperformance",
            target_col="forward_excess_return",
            min_bucket_size=min_bucket_size,
        ),
        _target_comparison_row(
            risk_adjusted_predictions,
            target_version="v2 risk-adjusted relative",
            target_col="forward_risk_adjusted_excess_return",
            min_bucket_size=min_bucket_size,
        ),
    ]
    if tail_risk_predictions is not None:
        rows.append(
            _target_comparison_row(
                tail_risk_predictions,
                target_version="v3 tail-risk relative",
                target_col="forward_tail_risk_adjusted_excess_return",
                min_bucket_size=min_bucket_size,
            )
        )
    comparison = pd.DataFrame(rows, columns=TARGET_COMPARISON_COLUMNS)
    if comparison.empty:
        return comparison

    reference = comparison.iloc[0]
    reference_score = _target_comparison_score(reference)
    reference_spread = reference_score[0] if reference_score is not None else pd.NA
    comparison.loc[comparison["target_version"] == "v1 outperformance", "relative_result"] = "v1 reference"
    comparison.loc[
        comparison["target_version"] == "v1 outperformance",
        "interpretation",
    ] = "Existing outperformance target used as the diagnostics reference."
    for target_version, version_label in (
        ("v2 risk-adjusted relative", "v2"),
        ("v3 tail-risk relative", "v3"),
    ):
        mask = comparison["target_version"] == target_version
        if not mask.any():
            continue
        row = comparison[mask].iloc[0]
        result = _target_comparison_result(reference, row, version_label)
        comparison.loc[mask, "relative_result"] = result
        comparison.loc[mask, "baseline_spread"] = reference_spread
        if result.endswith("looks better"):
            interpretation = (
                f"{version_label} has cleaner bucket separation or monotonicity without clearly "
                "worse spread than v1 in this sample."
            )
        elif result.endswith("looks worse"):
            interpretation = f"{version_label} has weaker bucket separation or monotonicity than v1 in this sample."
        elif result.endswith("looks similar"):
            interpretation = f"{version_label} and v1 show similar diagnostics in this sample."
        else:
            interpretation = f"The sample is too small or incomplete for a v1-vs-{version_label} read."
        comparison.loc[mask, "interpretation"] = interpretation
    return comparison


def build_score_inversion_diagnostics(
    score_panel: pd.DataFrame,
    *,
    score_col: str = "ML Score",
    target_col: str = "forward_excess_return",
    min_bucket_size: int = MIN_SCORE_DIRECTION_BUCKET_COUNT,
) -> pd.DataFrame:
    """Compare current score buckets with an inverted score used only inside this diagnostic."""

    if score_panel.empty or score_col not in score_panel or target_col not in score_panel:
        return _empty_score_inversion_diagnostics()

    data = score_panel[[score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return _empty_score_inversion_diagnostics()

    score_max = data[score_col].max()
    inverted_max = 100.0 if pd.notna(score_max) and score_max > 1.0 else 1.0
    data = data.copy()
    data["_inverted_score"] = inverted_max - data[score_col]

    rows: list[dict[str, object]] = []
    for direction_label, column in (("current ML score", score_col), ("inverted score", "_inverted_score")):
        sample_size, top_return, bottom_return, spread = _bucket_spread(
            data,
            column,
            target_col,
            min_bucket_size,
        )
        rows.append(
            {
                "score_direction": direction_label,
                "sample_size": sample_size,
                "top_bucket_forward_return": top_return,
                "bottom_bucket_forward_return": bottom_return,
                "top_minus_bottom_spread": spread,
                "better_forward_return_separation": "",
                "interpretation": "",
            }
        )

    current_spread = pd.to_numeric(pd.Series([rows[0]["top_minus_bottom_spread"]]), errors="coerce").iloc[0]
    inverted_spread = pd.to_numeric(pd.Series([rows[1]["top_minus_bottom_spread"]]), errors="coerce").iloc[0]
    if pd.isna(current_spread) or pd.isna(inverted_spread):
        better = "insufficient"
        interpretation = "The score buckets are too small for an inversion comparison."
    elif current_spread > inverted_spread + SPREAD_SIMILARITY_TOLERANCE:
        better = "current ML score"
        interpretation = "The current score direction gives better forward-excess-return separation in this sample."
    elif inverted_spread > current_spread + SPREAD_SIMILARITY_TOLERANCE:
        better = "inverted score"
        interpretation = "The inverted score direction gives better forward-excess-return separation in this sample."
    else:
        better = "similar"
        interpretation = "Current and inverted score directions show similar forward-excess-return separation."

    for row in rows:
        row["better_forward_return_separation"] = better
        row["interpretation"] = interpretation

    return pd.DataFrame(rows, columns=SCORE_INVERSION_COLUMNS)


def _probability_direction_candidate_row(
    data: pd.DataFrame,
    *,
    signal: str,
    signal_col: str,
    return_col: str,
    label_col: str,
    min_bucket_size: int,
    min_samples: int,
) -> dict[str, object]:
    empty_row = {
        "signal": signal,
        "sample_size": 0,
        "low_bucket_forward_excess_return": pd.NA,
        "mid_bucket_forward_excess_return": pd.NA,
        "high_bucket_forward_excess_return": pd.NA,
        "high_minus_low_spread": pd.NA,
        "monotonicity": "insufficient",
        "actual_label_rate_low_bucket": pd.NA,
        "actual_label_rate_high_bucket": pd.NA,
        "interpretation": "There is too little usable data for this probability direction check.",
    }
    if signal_col not in data or return_col not in data:
        return empty_row

    columns = [signal_col, return_col]
    has_label = label_col in data
    if has_label:
        columns.append(label_col)

    usable = data[columns].apply(pd.to_numeric, errors="coerce").dropna(subset=[signal_col, return_col])
    if (
        len(usable) < min_samples
        or usable[signal_col].nunique(dropna=True) < 3
    ):
        row = empty_row.copy()
        row["sample_size"] = int(len(usable))
        return row

    bucketed = usable.copy()
    try:
        bucketed["_direction_bucket"] = pd.qcut(
            bucketed[signal_col].rank(method="first"),
            q=3,
            labels=["Low", "Mid", "High"],
        )
    except ValueError:
        row = empty_row.copy()
        row["sample_size"] = int(len(usable))
        row["interpretation"] = "Probability buckets were too narrow for this direction check."
        return row

    grouped = bucketed.groupby("_direction_bucket", observed=True)
    bucket_returns: dict[str, object] = {}
    label_rates: dict[str, object] = {}
    for bucket in ("Low", "Mid", "High"):
        if bucket not in grouped.groups:
            bucket_returns[bucket] = pd.NA
            label_rates[bucket] = pd.NA
            continue
        group = grouped.get_group(bucket)
        if len(group) < min_bucket_size:
            bucket_returns[bucket] = pd.NA
            label_rates[bucket] = pd.NA
            continue
        bucket_returns[bucket] = float(group[return_col].mean())
        label_rates[bucket] = float(group[label_col].mean()) if has_label else pd.NA

    low_return = bucket_returns["Low"]
    mid_return = bucket_returns["Mid"]
    high_return = bucket_returns["High"]
    if pd.isna(low_return) or pd.isna(high_return):
        spread = pd.NA
        monotonicity = "insufficient"
        interpretation = "Probability buckets were too small for this direction check."
    else:
        spread = float(high_return - low_return)
        complete_returns = [
            float(value)
            for value in (low_return, mid_return, high_return)
            if pd.notna(value)
        ]
        monotonicity = _monotonicity_result(complete_returns)
        if monotonicity == "aligned":
            interpretation = f"Higher {signal} buckets have higher realised forward excess return in this sample."
        elif monotonicity == "inverted":
            interpretation = f"Higher {signal} buckets have lower realised forward excess return in this sample."
        elif monotonicity == "flat":
            interpretation = f"{signal} buckets have similar realised forward excess return in this sample."
        else:
            interpretation = f"{signal} buckets have mixed realised forward excess return in this sample."

    return {
        "signal": signal,
        "sample_size": int(len(usable)),
        "low_bucket_forward_excess_return": low_return,
        "mid_bucket_forward_excess_return": mid_return,
        "high_bucket_forward_excess_return": high_return,
        "high_minus_low_spread": spread,
        "monotonicity": monotonicity,
        "actual_label_rate_low_bucket": label_rates["Low"],
        "actual_label_rate_high_bucket": label_rates["High"],
        "interpretation": interpretation,
    }


def build_ml_probability_direction_check(
    score_panel: pd.DataFrame,
    *,
    probability_col: str = "probability_out",
    risk_probability_col: str = "probability_risk",
    score_col: str = "ML Score",
    return_col: str = "forward_excess_return",
    label_col: str = "actual_out",
    min_bucket_size: int = MIN_SCORE_DIRECTION_BUCKET_COUNT,
    min_samples: int = MIN_SCORE_DIRECTION_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Compare raw, inverted, and current-score directions against realised excess returns."""

    if score_panel.empty or probability_col not in score_panel or return_col not in score_panel:
        return _empty_ml_probability_direction_check()

    data = score_panel.copy()
    data["raw_outperform_probability"] = _numeric_column(data, probability_col)
    data["inverted_outperform_probability"] = 1.0 - data["raw_outperform_probability"]
    candidate_specs = [
        ("raw probability", "raw_outperform_probability"),
        ("inverted probability", "inverted_outperform_probability"),
    ]
    if score_col in data:
        data["current_ml_score"] = _numeric_column(data, score_col)
        candidate_specs.append(("current ML Score", "current_ml_score"))
    elif risk_probability_col in data:
        data["current_ml_score"] = ml_score(
            data["raw_outperform_probability"],
            _numeric_column(data, risk_probability_col),
        )
        candidate_specs.append(("current ML Score", "current_ml_score"))

    rows = [
        _probability_direction_candidate_row(
            data,
            signal=signal,
            signal_col=column,
            return_col=return_col,
            label_col=label_col,
            min_bucket_size=min_bucket_size,
            min_samples=min_samples,
        )
        for signal, column in candidate_specs
    ]
    return pd.DataFrame(rows, columns=ML_PROBABILITY_DIRECTION_COLUMNS)


def _risk_haircut_score(opportunity_score: pd.Series, risk_probability: pd.Series) -> pd.Series:
    haircut = pd.Series(1.0, index=opportunity_score.index)
    haircut = haircut.mask(risk_probability >= 0.40, 0.85)
    haircut = haircut.mask(risk_probability >= 0.60, 0.70)
    return (opportunity_score * haircut).clip(lower=0.0, upper=100.0)


def _score_formula_interpretation(
    *,
    spread: object,
    monotonicity: str,
    low_label_rate: object,
    high_label_rate: object,
    low_drawdown_event_rate: object,
    high_drawdown_event_rate: object,
) -> str:
    if pd.isna(spread) or monotonicity == "insufficient":
        return "insufficient data"
    if float(spread) < -SPREAD_SIMILARITY_TOLERANCE or monotonicity == "inverted":
        return "candidate is inverted"

    label_support = True
    if pd.notna(low_label_rate) and pd.notna(high_label_rate):
        label_support = high_label_rate > low_label_rate + SPREAD_SIMILARITY_TOLERANCE

    drawdown_acceptable = True
    if pd.notna(low_drawdown_event_rate) and pd.notna(high_drawdown_event_rate):
        drawdown_acceptable = high_drawdown_event_rate <= low_drawdown_event_rate + 0.20

    if (
        float(spread) > SPREAD_SIMILARITY_TOLERANCE
        and monotonicity == "aligned"
        and label_support
        and drawdown_acceptable
    ):
        return "candidate looks strong"
    if float(spread) <= SPREAD_SIMILARITY_TOLERANCE or not label_support:
        return "candidate looks weak"
    return "candidate is mixed"


def _formula_candidate_row(
    data: pd.DataFrame,
    *,
    candidate_name: str,
    score_col: str,
    return_col: str,
    label_col: str,
    drawdown_event_col: str | None,
    min_bucket_size: int,
    min_samples: int,
) -> dict[str, object]:
    empty_row = {
        "candidate_name": candidate_name,
        "sample_size": 0,
        "low_bucket_forward_excess_return": pd.NA,
        "mid_bucket_forward_excess_return": pd.NA,
        "high_bucket_forward_excess_return": pd.NA,
        "high_minus_low_spread": pd.NA,
        "monotonicity": "insufficient",
        "low_bucket_label_rate": pd.NA,
        "high_bucket_label_rate": pd.NA,
        "low_bucket_drawdown_event_rate": pd.NA,
        "high_bucket_drawdown_event_rate": pd.NA,
        "interpretation": "insufficient data",
    }
    if score_col not in data or return_col not in data:
        return empty_row

    columns = [score_col, return_col]
    has_label = label_col in data
    if has_label:
        columns.append(label_col)
    has_drawdown_event = drawdown_event_col is not None and drawdown_event_col in data
    if has_drawdown_event:
        columns.append(drawdown_event_col)

    usable = data[columns].apply(pd.to_numeric, errors="coerce").dropna(subset=[score_col, return_col])
    if len(usable) < min_samples or usable[score_col].nunique(dropna=True) < 3:
        row = empty_row.copy()
        row["sample_size"] = int(len(usable))
        return row

    bucketed = usable.copy()
    try:
        bucketed["_candidate_bucket"] = pd.qcut(
            bucketed[score_col].rank(method="first"),
            q=3,
            labels=["Low", "Mid", "High"],
        )
    except ValueError:
        row = empty_row.copy()
        row["sample_size"] = int(len(usable))
        return row

    grouped = bucketed.groupby("_candidate_bucket", observed=True)
    bucket_returns: dict[str, object] = {}
    label_rates: dict[str, object] = {}
    drawdown_event_rates: dict[str, object] = {}
    for bucket in ("Low", "Mid", "High"):
        if bucket not in grouped.groups:
            bucket_returns[bucket] = pd.NA
            label_rates[bucket] = pd.NA
            drawdown_event_rates[bucket] = pd.NA
            continue
        group = grouped.get_group(bucket)
        if len(group) < min_bucket_size:
            bucket_returns[bucket] = pd.NA
            label_rates[bucket] = pd.NA
            drawdown_event_rates[bucket] = pd.NA
            continue
        bucket_returns[bucket] = float(group[return_col].mean())
        label_rates[bucket] = float(group[label_col].mean()) if has_label else pd.NA
        drawdown_event_rates[bucket] = (
            float(group[drawdown_event_col].mean()) if has_drawdown_event and drawdown_event_col else pd.NA
        )

    low_return = bucket_returns["Low"]
    mid_return = bucket_returns["Mid"]
    high_return = bucket_returns["High"]
    if pd.isna(low_return) or pd.isna(mid_return) or pd.isna(high_return):
        spread = pd.NA
        monotonicity = "insufficient"
    else:
        spread = float(high_return - low_return)
        monotonicity = _monotonicity_result([float(low_return), float(mid_return), float(high_return)])

    interpretation = _score_formula_interpretation(
        spread=spread,
        monotonicity=monotonicity,
        low_label_rate=label_rates["Low"],
        high_label_rate=label_rates["High"],
        low_drawdown_event_rate=drawdown_event_rates["Low"],
        high_drawdown_event_rate=drawdown_event_rates["High"],
    )
    return {
        "candidate_name": candidate_name,
        "sample_size": int(len(usable)),
        "low_bucket_forward_excess_return": low_return,
        "mid_bucket_forward_excess_return": mid_return,
        "high_bucket_forward_excess_return": high_return,
        "high_minus_low_spread": spread,
        "monotonicity": monotonicity,
        "low_bucket_label_rate": label_rates["Low"],
        "high_bucket_label_rate": label_rates["High"],
        "low_bucket_drawdown_event_rate": drawdown_event_rates["Low"],
        "high_bucket_drawdown_event_rate": drawdown_event_rates["High"],
        "interpretation": interpretation,
    }


def build_ml_score_formula_candidate_comparison(
    score_panel: pd.DataFrame,
    *,
    probability_col: str = "probability_out",
    risk_probability_col: str = "probability_risk",
    return_col: str = "forward_excess_return",
    label_col: str = "actual_out",
    drawdown_label_col: str = "actual_risk",
    forward_drawdown_col: str = "forward_drawdown",
    drawdown_event_threshold: float = -0.10,
    min_bucket_size: int = MIN_SCORE_DIRECTION_BUCKET_COUNT,
    min_samples: int = MIN_SCORE_DIRECTION_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Compare diagnostics-only ML score formula candidates on one validation panel."""

    if score_panel.empty or probability_col not in score_panel or risk_probability_col not in score_panel:
        return _empty_ml_formula_candidate_comparison()
    if return_col not in score_panel:
        return _empty_ml_formula_candidate_comparison()

    data = score_panel.copy()
    probability = _numeric_column(data, probability_col).clip(lower=0.0, upper=1.0)
    risk_probability = _numeric_column(data, risk_probability_col).clip(lower=0.0, upper=1.0)
    sqrt_opportunity = probability.pow(0.5)

    data["_raw_probability_score"] = 100.0 * probability
    data["_sqrt_opportunity_only_score"] = 100.0 * sqrt_opportunity
    data["_current_production_score"] = ml_score(probability, risk_probability)
    data["_light_risk_penalty_score"] = 100.0 * (0.85 * sqrt_opportunity + 0.15 * (1.0 - risk_probability))
    data["_risk_haircut_score"] = _risk_haircut_score(data["_sqrt_opportunity_only_score"], risk_probability)
    drawdown_event_col = None
    if drawdown_label_col in data:
        drawdown_event_col = drawdown_label_col
    elif forward_drawdown_col in data:
        drawdown_event_col = "_drawdown_event"
        data[drawdown_event_col] = (_numeric_column(data, forward_drawdown_col) <= drawdown_event_threshold).astype(float)

    candidate_specs = [
        ("raw_probability", "_raw_probability_score"),
        ("sqrt_opportunity_only", "_sqrt_opportunity_only_score"),
        ("current_production_score", "_current_production_score"),
        ("light_risk_penalty", "_light_risk_penalty_score"),
        ("risk_haircut_score", "_risk_haircut_score"),
    ]
    rows = [
        _formula_candidate_row(
            data,
            candidate_name=candidate_name,
            score_col=score_col,
            return_col=return_col,
            label_col=label_col,
            drawdown_event_col=drawdown_event_col,
            min_bucket_size=min_bucket_size,
            min_samples=min_samples,
        )
        for candidate_name, score_col in candidate_specs
    ]
    return pd.DataFrame(rows, columns=ML_FORMULA_CANDIDATE_COLUMNS)


def interpret_ml_score_formula_candidate_comparison(comparison: pd.DataFrame) -> str:
    """Return one conservative conclusion for the formula candidate comparison."""

    if comparison.empty or "candidate_name" not in comparison:
        return "insufficient data"
    usable = comparison[comparison["interpretation"] != "insufficient data"].copy()
    if usable.empty:
        return "insufficient data"

    strong = usable[usable["interpretation"] == "candidate looks strong"].copy()
    if strong.empty:
        return "formula evidence is mixed"
    strong["high_minus_low_spread"] = pd.to_numeric(strong["high_minus_low_spread"], errors="coerce")
    strong = strong.dropna(subset=["high_minus_low_spread"]).sort_values(
        ["high_minus_low_spread", "candidate_name"],
        ascending=[False, True],
    )
    if strong.empty:
        return "formula evidence is mixed"

    def candidate_group(name: str) -> str:
        if name in {"raw_probability", "sqrt_opportunity_only"}:
            return "opportunity_only"
        return name

    best = strong.iloc[0]
    best_spread = float(best["high_minus_low_spread"])
    near_best = strong[strong["high_minus_low_spread"] >= best_spread - SPREAD_SIMILARITY_TOLERANCE]
    near_best_groups = {candidate_group(str(name)) for name in near_best["candidate_name"]}
    if len(near_best_groups) > 1:
        return "formula evidence is mixed"

    candidate_name = str(best["candidate_name"])
    if candidate_name in {"raw_probability", "sqrt_opportunity_only"}:
        return "opportunity-only scoring looks strongest"
    if candidate_name == "light_risk_penalty":
        return "light risk penalty looks strongest"
    if candidate_name == "risk_haircut_score":
        return "risk haircut scoring looks strongest"
    if candidate_name == "current_production_score":
        return "current production score looks strongest"
    return "formula evidence is mixed"


def _probability_direction_row_supported(row: pd.Series, *, require_label: bool) -> bool:
    if row.get("monotonicity") != "aligned":
        return False
    spread = pd.to_numeric(pd.Series([row.get("high_minus_low_spread")]), errors="coerce").iloc[0]
    if pd.isna(spread) or spread <= SPREAD_SIMILARITY_TOLERANCE:
        return False
    if not require_label:
        return True
    label_low = pd.to_numeric(pd.Series([row.get("actual_label_rate_low_bucket")]), errors="coerce").iloc[0]
    label_high = pd.to_numeric(pd.Series([row.get("actual_label_rate_high_bucket")]), errors="coerce").iloc[0]
    return pd.notna(label_low) and pd.notna(label_high) and label_high > label_low + SPREAD_SIMILARITY_TOLERANCE


def interpret_ml_probability_direction_check(direction_check: pd.DataFrame) -> str:
    """Return one compact conclusion for the probability direction check."""

    if direction_check.empty or "signal" not in direction_check:
        return "insufficient data"

    rows = direction_check.set_index("signal")
    raw_supported = (
        "raw probability" in rows.index
        and _probability_direction_row_supported(rows.loc["raw probability"], require_label=True)
    )
    inverted_supported = (
        "inverted probability" in rows.index
        and _probability_direction_row_supported(rows.loc["inverted probability"], require_label=True)
    )
    score_supported = (
        "current ML Score" in rows.index
        and _probability_direction_row_supported(rows.loc["current ML Score"], require_label=False)
    )

    if raw_supported and not inverted_supported:
        return "raw outperformance probability direction is supported"
    if inverted_supported and not raw_supported:
        return "inverted outperformance probability direction is supported"
    if score_supported and not raw_supported and not inverted_supported:
        return "current ML Score direction is supported"

    monotonicity = set(direction_check["monotonicity"].dropna().astype(str))
    if not monotonicity or monotonicity == {"insufficient"}:
        return "insufficient data"
    return "direction evidence is mixed"


def build_regime_score_direction_summary(
    score_panel: pd.DataFrame,
    *,
    baseline_panel: pd.DataFrame | None = None,
    score_col: str = "ML Score",
    target_col: str = "forward_excess_return",
    regime_cols: list[str] | None = None,
    min_bucket_size: int = MIN_SCORE_DIRECTION_BUCKET_COUNT,
    min_samples: int = MIN_REGIME_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Summarize existing score direction by available regime labels."""

    if score_panel.empty:
        return _empty_regime_score_direction_summary()

    candidate_regime_cols = list(regime_cols or DEFAULT_REGIME_COLUMNS)
    data = _merge_regime_panel(score_panel, baseline_panel, candidate_regime_cols)
    if score_col not in data or target_col not in data:
        return _empty_regime_score_direction_summary()

    available_regime_cols = [column for column in candidate_regime_cols if column in data]
    if not available_regime_cols:
        return _empty_regime_score_direction_summary()

    data = data.copy()
    data[score_col] = _numeric_column(data, score_col)
    data[target_col] = _numeric_column(data, target_col)
    data = data.dropna(subset=[score_col, target_col])
    if data.empty:
        return _empty_regime_score_direction_summary()

    rows: list[dict[str, object]] = []
    for regime_col in available_regime_cols:
        regime_data = data.dropna(subset=[regime_col]).copy()
        if regime_data.empty:
            continue
        regime_data[regime_col] = regime_data[regime_col].astype(str).str.strip()
        regime_data = regime_data[regime_data[regime_col] != ""]
        for regime, group in regime_data.groupby(regime_col, sort=True):
            sample_size, top_return, bottom_return, spread = _bucket_spread(
                group,
                score_col,
                target_col,
                min_bucket_size,
            )
            direction = _score_direction(spread, sample_size, min_samples)
            rows.append(
                {
                    "regime_dimension": regime_col,
                    "regime": regime,
                    "sample_size": sample_size,
                    "top_bucket_forward_return": top_return,
                    "bottom_bucket_forward_return": bottom_return,
                    "top_minus_bottom_spread": spread,
                    "direction": direction,
                    "interpretation": _score_direction_interpretation(direction),
                }
            )

    if not rows:
        return _empty_regime_score_direction_summary()
    return pd.DataFrame(rows, columns=REGIME_SCORE_DIRECTION_COLUMNS)


def build_ml_diagnostics(
    outperformance_predictions: pd.DataFrame,
    drawdown_risk_predictions: pd.DataFrame,
    outperformance_metrics: pd.DataFrame | None = None,
    drawdown_risk_metrics: pd.DataFrame | None = None,
    risk_bins: int = 5,
    baseline_panel: pd.DataFrame | None = None,
    risk_adjusted_predictions: pd.DataFrame | None = None,
    tail_risk_predictions: pd.DataFrame | None = None,
) -> MLDiagnostics:
    """Summarize out-of-sample usefulness of the existing ML signal.

    Inputs should come from existing walk-forward validation results. This helper
    only joins predictions and summarizes diagnostics; it does not fit models or
    change score, probability, or decision logic.
    """

    out_columns = [
        "Date",
        "Ticker",
        "actual",
        "probability",
        "forward_return",
        "forward_excess_return",
        "forward_drawdown",
    ]
    if "fold" in outperformance_predictions:
        out_columns.insert(0, "fold")
    risk_columns = ["Date", "Ticker", "actual", "probability"]
    if "fold" in drawdown_risk_predictions:
        risk_columns.insert(0, "fold")
    out = outperformance_predictions[
        [column for column in out_columns if column in outperformance_predictions]
    ].rename(columns={"actual": "actual_out", "probability": "probability_out"})
    risk = drawdown_risk_predictions[
        [column for column in risk_columns if column in drawdown_risk_predictions]
    ].rename(
        columns={"actual": "actual_risk", "probability": "probability_risk"}
    )
    probability_direction_panel = out.copy()
    if (
        not probability_direction_panel.empty
        and not risk.empty
        and {"Date", "Ticker"}.issubset(probability_direction_panel.columns)
        and {"Date", "Ticker"}.issubset(risk.columns)
    ):
        keys = prediction_merge_keys(probability_direction_panel, risk)
        probability_direction_panel = deduplicate_prediction_keys(
            probability_direction_panel,
            keys,
        ).merge(
            deduplicate_prediction_keys(risk, keys),
            on=keys,
            how="left",
        )
    if not probability_direction_panel.empty and "probability_risk" in probability_direction_panel:
        probability_direction_panel["ML Score"] = ml_score(
            probability_direction_panel["probability_out"],
            probability_direction_panel["probability_risk"],
        )
    probability_direction_check = build_ml_probability_direction_check(probability_direction_panel)
    formula_candidate_comparison = build_ml_score_formula_candidate_comparison(probability_direction_panel)

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
            _empty_score_direction_summary(),
            _empty_probability_label_alignment(),
            _empty_score_bucket_monotonicity(),
            _empty_score_inversion_diagnostics(),
            _empty_regime_score_direction_summary(),
            _empty_drawdown_risk_calibration_quality(),
            _empty_target_comparison(),
            _empty_opportunity_risk_joint_validation(),
            probability_direction_check,
            formula_candidate_comparison,
        )

    keys = prediction_merge_keys(out, risk)
    score_panel = deduplicate_prediction_keys(out, keys).merge(
        deduplicate_prediction_keys(risk, keys),
        on=keys,
        how="inner",
    )
    if not score_panel.empty:
        score_panel["ML Score"] = ml_score(score_panel["probability_out"], score_panel["probability_risk"])
    baseline_comparison = build_ml_baseline_comparison(score_panel, baseline_panel)
    regime_segmented = build_regime_segmented_ml_diagnostics(score_panel, baseline_panel=baseline_panel)
    score_direction_summary = build_ml_score_direction_diagnostics(score_panel)
    probability_label_alignment = build_probability_label_alignment(score_panel)
    score_bucket_monotonicity = build_score_bucket_monotonicity(score_panel)
    score_inversion = build_score_inversion_diagnostics(score_panel)
    regime_score_direction = build_regime_score_direction_summary(score_panel, baseline_panel=baseline_panel)
    opportunity_risk_joint_validation = build_opportunity_risk_joint_validation(score_panel)
    target_comparison = (
        build_ml_target_comparison(
            outperformance_predictions,
            risk_adjusted_predictions,
            tail_risk_predictions,
        )
        if risk_adjusted_predictions is not None
        else _empty_target_comparison()
    )

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
        score_direction_summary,
        probability_label_alignment,
        score_bucket_monotonicity,
        score_inversion,
        regime_score_direction,
        risk_calibration_quality,
        target_comparison,
        opportunity_risk_joint_validation,
        probability_direction_check,
        formula_candidate_comparison,
    )
