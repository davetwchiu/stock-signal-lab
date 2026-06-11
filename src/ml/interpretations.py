"""Plain-language interpretations for existing ML diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MIN_ML_HEALTH_BUCKET_COUNT = 5


@dataclass(frozen=True)
class DiagnosticInterpretation:
    """Display-only interpretation of an existing diagnostics table."""

    label: str
    message: str
    level: str = "info"


@dataclass(frozen=True)
class ResearchLabRunInterpretation:
    """Display-only interpretation of an existing Research Lab run."""

    overall: str
    walk_forward_validation: str
    ml_score_buckets: str
    drawdown_risk_calibration: str
    use: str
    level: str = "info"


def _numeric_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _format_signed_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1%}"


def _diagnostic_bucket_row(score_buckets: pd.DataFrame, bucket: str) -> pd.Series | None:
    if score_buckets.empty or "score_bucket" not in score_buckets:
        return None
    bucket_rows = score_buckets[score_buckets["score_bucket"].astype(str) == bucket]
    return None if bucket_rows.empty else bucket_rows.iloc[0]


def _preferred_return_column(score_buckets: pd.DataFrame) -> str | None:
    for column in ("average_forward_excess_return", "average_forward_return"):
        if column in score_buckets:
            return column
    return None


def interpret_ml_score_buckets(score_buckets: pd.DataFrame) -> DiagnosticInterpretation:
    """Explain whether higher score buckets separated stronger forward outcomes."""

    low_bucket = _diagnostic_bucket_row(score_buckets, "Low")
    high_bucket = _diagnostic_bucket_row(score_buckets, "High")
    return_column = _preferred_return_column(score_buckets)
    if low_bucket is None or high_bucket is None or return_column is None:
        return DiagnosticInterpretation(
            "Insufficient sample",
            "The score-bucket table is missing low/high buckets or forward outcome columns, so this sample cannot support a firm read.",
        )

    low_count = _numeric_or_none(low_bucket.get("count"))
    high_count = _numeric_or_none(high_bucket.get("count"))
    if (
        low_count is None
        or high_count is None
        or low_count < MIN_ML_HEALTH_BUCKET_COUNT
        or high_count < MIN_ML_HEALTH_BUCKET_COUNT
    ):
        return DiagnosticInterpretation(
            "Insufficient sample",
            (
                "Low and high ML score buckets need more observations before the score can be read "
                "as useful ranking evidence."
            ),
        )

    high_hit = _numeric_or_none(high_bucket.get("outperformance_hit_rate"))
    low_hit = _numeric_or_none(low_bucket.get("outperformance_hit_rate"))
    high_return = _numeric_or_none(high_bucket.get(return_column))
    low_return = _numeric_or_none(low_bucket.get(return_column))
    if high_hit is None or low_hit is None or high_return is None or low_return is None:
        return DiagnosticInterpretation(
            "Insufficient sample",
            "The score buckets have missing hit-rate or forward-return values, so treat this diagnostic as incomplete.",
        )

    hit_spread = high_hit - low_hit
    return_spread = high_return - low_return
    hit_better = hit_spread > 0.02
    return_better = return_spread > 0
    hit_worse = hit_spread < -0.02
    return_worse = return_spread < 0

    if hit_better and return_better:
        return DiagnosticInterpretation(
            "Good separation",
            (
                "Higher ML score buckets show better forward outcomes than lower buckets in this sample. "
                "That supports using ML score as a secondary research input, not as a direct action signal."
            ),
            "success",
        )
    if hit_worse and return_worse:
        return DiagnosticInterpretation(
            "Treat ML score cautiously",
            (
                "Higher ML score buckets performed worse than lower buckets in this sample, so the score "
                "is weak ranking evidence here."
            ),
            "warning",
        )
    if abs(hit_spread) <= 0.02 and abs(return_spread) <= 0.005:
        return DiagnosticInterpretation(
            "Weak separation",
            (
                "High and low ML score buckets look similar in this sample, so the score is weak ranking evidence."
            ),
            "warning",
        )
    return DiagnosticInterpretation(
        "Mixed evidence",
        (
            "The score buckets separate on some outcomes but not others. Treat ML evidence cautiously "
            "and use it only as supporting research context."
        ),
    )


def interpret_drawdown_calibration(calibration: pd.DataFrame) -> DiagnosticInterpretation:
    """Explain whether predicted drawdown-risk buckets match observed drawdowns."""

    required = {"average_probability", "observed_drawdown_risk_rate", "count"}
    if calibration.empty or not required.issubset(calibration.columns):
        return DiagnosticInterpretation(
            "Insufficient sample",
            "The drawdown-risk calibration table is missing usable probability, observed-rate, or count values.",
        )

    data = calibration.dropna(subset=["average_probability", "observed_drawdown_risk_rate"]).copy()
    data = data[data["count"] > 0].sort_values("average_probability")
    if len(data) < 2 or data["count"].sum() < MIN_ML_HEALTH_BUCKET_COUNT * 2:
        return DiagnosticInterpretation(
            "Insufficient sample",
            "There are too few drawdown-risk calibration observations for a firm read.",
        )

    low_rate = _numeric_or_none(data.iloc[0]["observed_drawdown_risk_rate"])
    high_rate = _numeric_or_none(data.iloc[-1]["observed_drawdown_risk_rate"])
    if low_rate is None or high_rate is None:
        return DiagnosticInterpretation(
            "Insufficient sample",
            "Observed drawdown rates are missing, so the risk calibration read is incomplete.",
        )

    rate_spread = high_rate - low_rate
    if low_rate >= 0.50:
        return DiagnosticInterpretation(
            "Tail risk may be understated",
            (
                "Even the lowest predicted-risk bucket had frequent drawdown events. Treat low-risk "
                "readings as cautious evidence rather than comfort."
            ),
            "warning",
        )
    if rate_spread > 0.02:
        return DiagnosticInterpretation(
            "Risk calibration looks useful",
            (
                "Higher predicted-risk buckets also show more observed drawdown events in this sample, "
                "so the risk diagnostic has useful separation."
            ),
            "success",
        )
    if rate_spread < -0.02:
        return DiagnosticInterpretation(
            "Risk calibration is weak",
            (
                "Observed drawdown events do not rise with predicted risk in this sample. Treat the "
                "drawdown-risk diagnostic as weak evidence."
            ),
            "warning",
        )
    return DiagnosticInterpretation(
        "Risk calibration is weak",
        (
            "Observed drawdown outcomes are similar across predicted-risk buckets, so the calibration "
            "does not show useful separation in this sample."
        ),
        "warning",
    )


def interpret_ml_diagnostics_summary(summary: pd.DataFrame) -> DiagnosticInterpretation:
    """Explain how much coverage the existing diagnostics summary provides."""

    if summary.empty or "predictions" not in summary:
        return DiagnosticInterpretation(
            "Insufficient sample",
            "The diagnostics summary has no prediction coverage, so confidence in the research evidence should stay low.",
        )
    predictions = pd.to_numeric(summary["predictions"], errors="coerce").fillna(0)
    folds = pd.to_numeric(summary.get("folds", pd.Series([0] * len(summary))), errors="coerce").fillna(0)
    if predictions.min() < MIN_ML_HEALTH_BUCKET_COUNT * 2 or folds.max() < 2:
        return DiagnosticInterpretation(
            "Insufficient sample",
            (
                "The walk-forward diagnostics have limited prediction or fold coverage. Treat ML evidence "
                "as preliminary research context."
            ),
        )
    return DiagnosticInterpretation(
        "Research coverage available",
        (
            "The summary has enough walk-forward coverage to review the diagnostics, but it only affects "
            "confidence in the research evidence. It does not create buy or sell instructions."
        ),
    )


def _summary_target_row(summary: pd.DataFrame, target: str) -> pd.Series | None:
    if summary.empty or "target" not in summary:
        return None
    rows = summary[summary["target"].astype(str) == target]
    return None if rows.empty else rows.iloc[0]


def _walk_forward_validation_read(summary: pd.DataFrame) -> tuple[str, str]:
    row = _summary_target_row(summary, "outperformance")
    if row is None:
        return (
            "insufficient",
            "The walk-forward validation summary is missing, so this run is inconclusive.",
        )

    predictions = _numeric_or_none(row.get("predictions"))
    folds = _numeric_or_none(row.get("folds"))
    if (
        predictions is None
        or folds is None
        or predictions < MIN_ML_HEALTH_BUCKET_COUNT * 2
        or folds < 2
    ):
        return (
            "insufficient",
            "The walk-forward sample is too small for a firm read.",
        )

    roc_auc = _numeric_or_none(row.get("roc_auc"))
    f1 = _numeric_or_none(row.get("f1"))
    if roc_auc is None and f1 is None:
        return (
            "mixed",
            "The walk-forward run has enough coverage to review, but the headline validation metrics are incomplete.",
        )
    if (roc_auc is not None and roc_auc >= 0.58) and (f1 is None or f1 >= 0.45):
        return (
            "useful",
            "Walk-forward validation has usable directional evidence in this sample.",
        )
    if (roc_auc is not None and roc_auc < 0.50) and (f1 is not None and f1 < 0.35):
        return (
            "weak",
            "Walk-forward validation is weak in this sample.",
        )
    return (
        "mixed",
        "Walk-forward validation is mixed, so the run is useful for review but not strong evidence by itself.",
    )


def _score_bucket_read(score_buckets: pd.DataFrame) -> tuple[str, str]:
    interpretation = interpret_ml_score_buckets(score_buckets)
    if interpretation.label == "Good separation":
        return (
            "useful",
            "High-score buckets outperformed low-score buckets, so the ML score has useful ranking evidence in this sample.",
        )
    if interpretation.label == "Insufficient sample":
        return (
            "insufficient",
            "The ML score buckets have too little usable data for a firm read.",
        )
    if interpretation.label == "Weak separation":
        return (
            "weak",
            "High-score buckets did not clearly outperform low-score buckets, so the ML score has weak ranking evidence in this sample.",
        )
    if interpretation.label == "Treat ML score cautiously":
        return (
            "weak",
            "High-score buckets performed worse than low-score buckets, so the ML score has weak ranking evidence in this sample.",
        )
    return (
        "mixed",
        "ML score buckets separate on some outcomes but not others, so the ranking evidence is mixed.",
    )


def _drawdown_calibration_read(calibration: pd.DataFrame) -> tuple[str, str]:
    interpretation = interpret_drawdown_calibration(calibration)
    if interpretation.label == "Risk calibration looks useful":
        return (
            "useful",
            "Drawdown-risk buckets separate actual drawdowns reasonably well, so the risk signal is useful as a caution flag.",
        )
    if interpretation.label == "Insufficient sample":
        return (
            "insufficient",
            "The drawdown-risk calibration sample is too small or incomplete for a firm read.",
        )
    return (
        "weak",
        "Drawdown-risk buckets do not clearly separate actual drawdowns in this sample.",
    )


def interpret_research_lab_run(
    diagnostics_summary: pd.DataFrame,
    score_buckets: pd.DataFrame,
    drawdown_risk_calibration: pd.DataFrame,
) -> ResearchLabRunInterpretation:
    """Summarize an existing walk-forward diagnostics run without changing calculations."""

    validation_status, validation_message = _walk_forward_validation_read(diagnostics_summary)
    score_status, score_message = _score_bucket_read(score_buckets)
    drawdown_status, drawdown_message = _drawdown_calibration_read(drawdown_risk_calibration)
    statuses = [validation_status, score_status, drawdown_status]

    if "insufficient" in statuses:
        return ResearchLabRunInterpretation(
            overall="Inconclusive research evidence.",
            walk_forward_validation=validation_message,
            ml_score_buckets=score_message,
            drawdown_risk_calibration=drawdown_message,
            use=(
                "The sample is too small or incomplete for confidence. This does not change Decision Mode; "
                "treat the research evidence cautiously."
            ),
            level="warning",
        )
    if statuses.count("useful") == 3:
        return ResearchLabRunInterpretation(
            overall="Usable but not strong research evidence.",
            walk_forward_validation=validation_message,
            ml_score_buckets=score_message,
            drawdown_risk_calibration=drawdown_message,
            use=(
                "This supports using ML score and drawdown risk as secondary research evidence. "
                "This does not change Decision Mode."
            ),
            level="success",
        )
    if statuses.count("weak") >= 2:
        return ResearchLabRunInterpretation(
            overall="Weak research evidence.",
            walk_forward_validation=validation_message,
            ml_score_buckets=score_message,
            drawdown_risk_calibration=drawdown_message,
            use=(
                "The diagnostics are too weak for confidence. This does not change Decision Mode; "
                "treat the research evidence cautiously."
            ),
            level="warning",
        )
    return ResearchLabRunInterpretation(
        overall="Mixed research evidence.",
        walk_forward_validation=validation_message,
        ml_score_buckets=score_message,
        drawdown_risk_calibration=drawdown_message,
        use=(
            "Use ML score and drawdown risk only as secondary research context. "
            "This does not change Decision Mode."
        ),
    )


def _drawdown_risk_calibration_health(calibration: pd.DataFrame) -> str:
    interpretation = interpret_drawdown_calibration(calibration)
    if interpretation.label == "Risk calibration looks useful":
        return "Rises with predicted risk"
    if interpretation.label == "Risk calibration is weak" and interpretation.level == "warning":
        return "Unclear"
    if interpretation.label == "Insufficient sample":
        return "Insufficient data"
    return interpretation.label


def ml_signal_health_interpretation(diagnostics) -> tuple[str, str, pd.DataFrame]:
    """Summarize existing ML diagnostics without changing production logic."""

    score_buckets = diagnostics.score_buckets
    low_bucket = _diagnostic_bucket_row(score_buckets, "Low")
    high_bucket = _diagnostic_bucket_row(score_buckets, "High")
    calibration_health = _drawdown_risk_calibration_health(diagnostics.drawdown_risk_calibration)

    empty_metrics = pd.DataFrame(
        [
            {"Metric": "Top score bucket hit rate", "Value": "N/A"},
            {"Metric": "Bottom score bucket hit rate", "Value": "N/A"},
            {"Metric": "Top-minus-bottom hit-rate spread", "Value": "N/A"},
            {"Metric": "Top score bucket average forward return", "Value": "N/A"},
            {"Metric": "Bottom score bucket average forward return", "Value": "N/A"},
            {"Metric": "Return spread", "Value": "N/A"},
            {"Metric": "Drawdown-risk calibration", "Value": calibration_health},
        ]
    )
    if low_bucket is None or high_bucket is None:
        return (
            "Insufficient data",
            "The diagnostics-only score buckets are missing low or high bucket observations in this walk-forward sample.",
            empty_metrics,
        )

    low_count = _numeric_or_none(low_bucket.get("count"))
    high_count = _numeric_or_none(high_bucket.get("count"))
    top_hit_rate = _numeric_or_none(high_bucket.get("outperformance_hit_rate"))
    bottom_hit_rate = _numeric_or_none(low_bucket.get("outperformance_hit_rate"))
    top_return = _numeric_or_none(high_bucket.get("average_forward_return"))
    bottom_return = _numeric_or_none(low_bucket.get("average_forward_return"))

    hit_spread = (
        top_hit_rate - bottom_hit_rate
        if top_hit_rate is not None and bottom_hit_rate is not None
        else None
    )
    return_spread = (
        top_return - bottom_return
        if top_return is not None and bottom_return is not None
        else None
    )

    metrics = pd.DataFrame(
        [
            {"Metric": "Top score bucket hit rate", "Value": _format_percent(top_hit_rate)},
            {"Metric": "Bottom score bucket hit rate", "Value": _format_percent(bottom_hit_rate)},
            {"Metric": "Top-minus-bottom hit-rate spread", "Value": _format_signed_percent(hit_spread)},
            {"Metric": "Top score bucket average forward return", "Value": _format_percent(top_return)},
            {"Metric": "Bottom score bucket average forward return", "Value": _format_percent(bottom_return)},
            {"Metric": "Return spread", "Value": _format_signed_percent(return_spread)},
            {"Metric": "Drawdown-risk calibration", "Value": calibration_health},
        ]
    )

    if (
        low_count is None
        or high_count is None
        or low_count < MIN_ML_HEALTH_BUCKET_COUNT
        or high_count < MIN_ML_HEALTH_BUCKET_COUNT
    ):
        return (
            "Insufficient data",
            (
                "The low and high ML score buckets are too small for a useful diagnostics-only read "
                f"in this walk-forward sample (minimum {MIN_ML_HEALTH_BUCKET_COUNT} observations each)."
            ),
            metrics,
        )

    hit_positive = hit_spread is not None and hit_spread > 0
    return_positive = return_spread is not None and return_spread > 0
    if hit_positive and return_positive:
        verdict = "Healthy"
        reason = (
            "The high ML score bucket appears better than the low bucket on both hit rate and average "
            "forward return in this walk-forward sample."
        )
    elif hit_positive or return_positive:
        verdict = "Mixed"
        reason = (
            "The high ML score bucket appears better on one diagnostics-only outcome, but the other "
            "outcome is weak, noisy, or unavailable in this walk-forward sample."
        )
    else:
        verdict = "Weak"
        reason = (
            "The high ML score bucket does not appear to outperform the low bucket on hit rate or "
            "average forward return in this walk-forward sample."
        )

    if calibration_health == "Rises with predicted risk":
        reason += " Drawdown-risk calibration rises with predicted risk."
    elif calibration_health == "Looks inverted":
        reason += " Drawdown-risk calibration looks inverted, so treat the risk read as a caveat."
    elif calibration_health == "Unclear":
        reason += " Drawdown-risk calibration is unclear, so treat the risk read as a caveat."
    else:
        reason += " Drawdown-risk calibration has insufficient data."

    return verdict, reason, metrics
