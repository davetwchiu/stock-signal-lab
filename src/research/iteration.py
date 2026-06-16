"""Codex-safe comparison helpers for Research Lab iterations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


AUC_DETERIORATION_TOLERANCE = 0.005
BRIER_DETERIORATION_TOLERANCE = 0.02
CALIBRATION_GAP_DETERIORATION_TOLERANCE = 0.03
BUCKET_SPREAD_DETERIORATION_TOLERANCE = 0.005


OBJECTIVE_TEMPLATES: dict[str, dict[str, object]] = {
    "ml_target_research": {
        "recommendation": "Keep the production target unless explicitly instructed.",
        "success_criteria": [
            "bucket spread improves or remains positive",
            "calibration gap does not materially worsen",
            "regime inversion count does not increase",
            "Brier score does not materially worsen",
            "evidence is stable across feature groups or clearly marked dependent",
        ],
    },
    "feature_engineering_research": {
        "recommendation": "Treat feature changes as research-only until diagnostics are stable.",
        "success_criteria": [
            "outperformance AUC and PR-AUC are not materially worse",
            "bucket spread is not materially worse",
            "unstable, missing, or redundant feature warnings do not increase",
            "drawdown-risk calibration does not deteriorate",
        ],
    },
    "calibration_research": {
        "recommendation": "Prefer calibration stability over isolated rank-metric gains.",
        "success_criteria": [
            "absolute calibration gap improves or is not materially worse",
            "Brier score improves or is not materially worse",
            "bucket spread does not invert",
            "regime stability does not worsen",
        ],
    },
    "regime_reliability_research": {
        "recommendation": "Do not promote changes that only work in one regime.",
        "success_criteria": [
            "regime inversion count does not increase",
            "regime-sensitive count does not increase",
            "overall AUC, PR-AUC, and bucket spread are not materially worse",
            "feature-group consistency does not deteriorate",
        ],
    },
}


@dataclass(frozen=True)
class ResearchRunComparison:
    """Structured result for a baseline-vs-candidate Research Lab comparison."""

    overall_status: str
    reason: str
    metric_deltas: dict[str, float | None]
    warnings: list[str]
    recommendation: str
    objective: str

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_status": self.overall_status,
            "reason": self.reason,
            "metric_deltas": self.metric_deltas,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
            "objective": self.objective,
        }


def compare_research_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    objective: str = "ml_target_research",
) -> dict[str, object]:
    """Compare two exported Research Lab bundles conservatively."""

    baseline = _load_bundle(Path(baseline_dir))
    candidate = _load_bundle(Path(candidate_dir))
    warnings: list[str] = []
    if objective not in OBJECTIVE_TEMPLATES:
        warnings.append(f"Unknown objective {objective}; using conservative thresholds.")

    metrics = _collect_metric_deltas(baseline, candidate, warnings)
    if not metrics:
        return ResearchRunComparison(
            overall_status="insufficient_evidence",
            reason="No comparable Research Lab metrics were available.",
            metric_deltas={},
            warnings=warnings,
            recommendation="insufficient_evidence",
            objective=objective,
        ).to_dict()

    worse_reasons = _worse_reasons(metrics)
    mixed_reasons = _mixed_reasons(metrics)
    improved_reasons = _improved_reasons(metrics)

    if any(reason.startswith("bucket_spread_inverted") for reason in worse_reasons):
        status = "worse"
    elif worse_reasons:
        status = "worse"
    elif mixed_reasons:
        status = "mixed"
    elif improved_reasons:
        status = "improved"
    else:
        status = "insufficient_evidence"

    recommendation = {
        "improved": "commit_candidate",
        "mixed": "manual_review_required",
        "worse": "do_not_commit_algorithm_change",
        "insufficient_evidence": "insufficient_evidence",
    }[status]
    reason_parts = worse_reasons or mixed_reasons or improved_reasons or ["No material metric movement."]
    return ResearchRunComparison(
        overall_status=status,
        reason="; ".join(reason_parts),
        metric_deltas={key: value.get("delta") for key, value in metrics.items()},
        warnings=warnings,
        recommendation=recommendation,
        objective=objective,
    ).to_dict()


def _load_bundle(path: Path) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    for filename in (
        "target_quality_summary.csv",
        "target_walk_forward_comparison.csv",
        "target_feature_group_comparison.csv",
        "target_regime_comparison.csv",
        "ml_diagnostics_summary.csv",
        "drawdown_risk_calibration_quality.csv",
    ):
        bundle[filename] = _read_csv(path / filename)
    return bundle


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _collect_metric_deltas(
    baseline: Mapping[str, pd.DataFrame],
    candidate: Mapping[str, pd.DataFrame],
    warnings: list[str],
) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    base_target = _target_row(baseline["target_quality_summary.csv"])
    cand_target = _target_row(candidate["target_quality_summary.csv"])
    if base_target.empty or cand_target.empty:
        warnings.append("target_quality_summary.csv missing or not comparable")
        base_target = _target_row(baseline["target_walk_forward_comparison.csv"])
        cand_target = _target_row(candidate["target_walk_forward_comparison.csv"])

    _add_delta(metrics, "outperformance_roc_auc", base_target, cand_target, "overall_auc", "roc_auc")
    _add_delta(metrics, "outperformance_pr_auc", base_target, cand_target, "overall_pr_auc", "pr_auc")
    _add_delta(metrics, "outperformance_brier_score", base_target, cand_target, "overall_brier_score", "brier_score")
    _add_delta(metrics, "bucket_spread", base_target, cand_target, "overall_bucket_spread", "bucket_spread")
    _add_delta(metrics, "calibration_gap", base_target, cand_target, "overall_calibration_gap", "calibration_gap")
    base_summary_out = _summary_row(baseline["ml_diagnostics_summary.csv"], "outperformance")
    cand_summary_out = _summary_row(candidate["ml_diagnostics_summary.csv"], "outperformance")
    _add_delta_if_missing(metrics, "outperformance_roc_auc", base_summary_out, cand_summary_out, "roc_auc")
    _add_delta_if_missing(metrics, "outperformance_pr_auc", base_summary_out, cand_summary_out, "pr_auc")
    _add_delta_if_missing(metrics, "outperformance_brier_score", base_summary_out, cand_summary_out, "brier_score")
    base_summary_risk = _summary_row(baseline["ml_diagnostics_summary.csv"], "drawdown_risk")
    cand_summary_risk = _summary_row(candidate["ml_diagnostics_summary.csv"], "drawdown_risk")
    _add_delta(metrics, "drawdown_risk_roc_auc", base_summary_risk, cand_summary_risk, "roc_auc")
    _add_delta(metrics, "drawdown_risk_brier_score", base_summary_risk, cand_summary_risk, "brier_score")
    metrics["target_quality_status"] = {
        "baseline": _quality_score(_first_value(base_target, "overall_target_quality", "production_candidate_status")),
        "candidate": _quality_score(_first_value(cand_target, "overall_target_quality", "production_candidate_status")),
        "delta": None,
    }
    metrics["regime_inversion_count"] = {
        "baseline": float(_regime_inversion_count(baseline["target_regime_comparison.csv"])),
        "candidate": float(_regime_inversion_count(candidate["target_regime_comparison.csv"])),
        "delta": float(
            _regime_inversion_count(candidate["target_regime_comparison.csv"])
            - _regime_inversion_count(baseline["target_regime_comparison.csv"])
        ),
    }
    metrics["regime_sensitive_count"] = {
        "baseline": float(_regime_sensitive_count(baseline["target_quality_summary.csv"])),
        "candidate": float(_regime_sensitive_count(candidate["target_quality_summary.csv"])),
        "delta": float(
            _regime_sensitive_count(candidate["target_quality_summary.csv"])
            - _regime_sensitive_count(baseline["target_quality_summary.csv"])
        ),
    }
    metrics["feature_group_consistency"] = {
        "baseline": _consistency_score(_first_value(base_target, "feature_group_consistency")),
        "candidate": _consistency_score(_first_value(cand_target, "feature_group_consistency")),
        "delta": None,
    }

    base_drawdown = _first_row(baseline["drawdown_risk_calibration_quality.csv"])
    cand_drawdown = _first_row(candidate["drawdown_risk_calibration_quality.csv"])
    _add_delta_if_missing(metrics, "drawdown_risk_brier_score", base_drawdown, cand_drawdown, "brier_score")
    _add_delta(metrics, "drawdown_risk_calibration_gap", base_drawdown, cand_drawdown, "calibration_gap")
    return {key: value for key, value in metrics.items() if _has_values(value)}


def _add_delta(
    metrics: dict[str, dict[str, float | None]],
    name: str,
    baseline_row: pd.Series,
    candidate_row: pd.Series,
    *columns: str,
) -> None:
    base = _numeric_first(baseline_row, *columns)
    cand = _numeric_first(candidate_row, *columns)
    metrics[name] = {
        "baseline": base,
        "candidate": cand,
        "delta": None if base is None or cand is None else cand - base,
    }


def _add_delta_if_missing(
    metrics: dict[str, dict[str, float | None]],
    name: str,
    baseline_row: pd.Series,
    candidate_row: pd.Series,
    *columns: str,
) -> None:
    if name in metrics and _has_values(metrics[name]):
        return
    _add_delta(metrics, name, baseline_row, candidate_row, *columns)


def _worse_reasons(metrics: Mapping[str, Mapping[str, float | None]]) -> list[str]:
    reasons: list[str] = []
    if _delta(metrics, "outperformance_roc_auc") < -AUC_DETERIORATION_TOLERANCE:
        reasons.append("outperformance ROC AUC materially worsened")
    if _delta(metrics, "outperformance_pr_auc") < -AUC_DETERIORATION_TOLERANCE:
        reasons.append("outperformance PR AUC materially worsened")
    if _delta(metrics, "outperformance_brier_score") > BRIER_DETERIORATION_TOLERANCE:
        reasons.append("outperformance Brier score materially worsened")
    if _delta(metrics, "bucket_spread") < -BUCKET_SPREAD_DETERIORATION_TOLERANCE:
        reasons.append("bucket spread materially worsened")
    candidate_spread = metrics.get("bucket_spread", {}).get("candidate")
    if candidate_spread is not None and candidate_spread < 0:
        reasons.append("bucket_spread_inverted: candidate bucket spread is negative")
    if _delta(metrics, "regime_inversion_count") > 0:
        reasons.append("regime inversion count increased")
    if _delta(metrics, "drawdown_risk_brier_score") > BRIER_DETERIORATION_TOLERANCE:
        reasons.append("drawdown-risk Brier score materially worsened")
    return reasons


def _mixed_reasons(metrics: Mapping[str, Mapping[str, float | None]]) -> list[str]:
    reasons: list[str] = []
    if _delta(metrics, "calibration_gap") > CALIBRATION_GAP_DETERIORATION_TOLERANCE:
        reasons.append("calibration gap worsened despite other evidence")
    if _delta(metrics, "drawdown_risk_calibration_gap") > CALIBRATION_GAP_DETERIORATION_TOLERANCE:
        reasons.append("drawdown-risk calibration worsened")
    if _delta(metrics, "regime_sensitive_count") > 0:
        reasons.append("regime-sensitive target count increased")
    fg = metrics.get("feature_group_consistency", {})
    if fg.get("candidate") is not None and fg.get("baseline") is not None and fg["candidate"] < fg["baseline"]:
        reasons.append("feature-group consistency worsened")
    quality = metrics.get("target_quality_status", {})
    if quality.get("candidate") is not None and quality.get("baseline") is not None and quality["candidate"] < quality["baseline"]:
        reasons.append("target quality status worsened")
    return reasons


def _improved_reasons(metrics: Mapping[str, Mapping[str, float | None]]) -> list[str]:
    reasons: list[str] = []
    if _delta(metrics, "outperformance_roc_auc") > AUC_DETERIORATION_TOLERANCE:
        reasons.append("outperformance ROC AUC improved")
    if _delta(metrics, "outperformance_pr_auc") > AUC_DETERIORATION_TOLERANCE:
        reasons.append("outperformance PR AUC improved")
    if _delta(metrics, "outperformance_brier_score") < -BRIER_DETERIORATION_TOLERANCE:
        reasons.append("outperformance Brier score improved")
    if _delta(metrics, "bucket_spread") > BUCKET_SPREAD_DETERIORATION_TOLERANCE:
        reasons.append("bucket spread improved")
    return reasons


def _target_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    if "target_id" in frame:
        mask = frame["target_id"].astype(str).str.contains("outperform_20d", case=False, na=False)
        if mask.any():
            return frame[mask].iloc[0]
    return frame.iloc[0]


def _summary_row(frame: pd.DataFrame, target: str) -> pd.Series:
    if frame.empty or "target" not in frame:
        return pd.Series(dtype=object)
    rows = frame[frame["target"].astype(str) == target]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _first_row(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)


def _numeric_first(row: pd.Series, *columns: str) -> float | None:
    for column in columns:
        if column in row and pd.notna(row[column]):
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def _first_value(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return ""


def _regime_inversion_count(frame: pd.DataFrame) -> int:
    if frame.empty or "direction" not in frame:
        return 0
    return int(frame["direction"].astype(str).str.contains("invert|negative", case=False, na=False).sum())


def _regime_sensitive_count(frame: pd.DataFrame) -> int:
    if frame.empty or "regime_stability" not in frame:
        return 0
    return int(frame["regime_stability"].astype(str).str.contains("sensitive|unstable|mixed", case=False, na=False).sum())


def _quality_score(value: str) -> float | None:
    text = value.lower()
    if not text:
        return None
    if "promising" in text or "trial" in text:
        return 4.0
    if "usable" in text or "keep" in text:
        return 3.0
    if "mixed" in text:
        return 2.0
    if "weak" in text or "special" in text:
        return 1.0
    if "unusable" in text or "reject" in text:
        return 0.0
    return None


def _consistency_score(value: str) -> float | None:
    text = value.lower()
    if not text:
        return None
    if "consistent positive" in text:
        return 4.0
    if "mostly positive" in text:
        return 3.0
    if "dependent" in text:
        return 2.0
    if "weak" in text:
        return 1.0
    if "inverted" in text or "unstable" in text:
        return 0.0
    return None


def _delta(metrics: Mapping[str, Mapping[str, float | None]], key: str) -> float:
    value = metrics.get(key, {}).get("delta")
    return float(value) if value is not None else 0.0


def _has_values(metric: Mapping[str, float | None]) -> bool:
    return metric.get("baseline") is not None or metric.get("candidate") is not None or metric.get("delta") is not None
