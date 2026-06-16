from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.compare_runs import main as compare_runs_main
from src.research.iteration import OBJECTIVE_TEMPLATES, compare_research_runs


def write_target_quality(
    run_dir: Path,
    *,
    auc: float,
    pr_auc: float,
    brier: float,
    calibration_gap: float,
    bucket_spread: float,
    quality: str = "Usable",
    feature_group_consistency: str = "Consistent positive",
    regime_stability: str = "Stable",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target_id": ["outperform_20d"],
            "overall_auc": [auc],
            "overall_pr_auc": [pr_auc],
            "overall_brier_score": [brier],
            "overall_calibration_gap": [calibration_gap],
            "overall_bucket_spread": [bucket_spread],
            "overall_target_quality": [quality],
            "feature_group_consistency": [feature_group_consistency],
            "regime_stability": [regime_stability],
        }
    ).to_csv(run_dir / "target_quality_summary.csv", index=False)


def write_regime(run_dir: Path, directions: list[str]) -> None:
    pd.DataFrame(
        {
            "target_id": ["outperform_20d"] * len(directions),
            "regime": [f"r{index}" for index, _ in enumerate(directions)],
            "direction": directions,
        }
    ).to_csv(run_dir / "target_regime_comparison.csv", index=False)


def write_ml_summary(run_dir: Path, *, drawdown_auc: float, drawdown_brier: float) -> None:
    pd.DataFrame(
        {
            "target": ["outperformance", "drawdown_risk"],
            "roc_auc": [0.60, drawdown_auc],
            "pr_auc": [0.54, 0.25],
            "brier_score": [0.22, drawdown_brier],
        }
    ).to_csv(run_dir / "ml_diagnostics_summary.csv", index=False)


def test_compare_research_runs_returns_improved_for_clear_gain(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_target_quality(baseline, auc=0.60, pr_auc=0.54, brier=0.22, calibration_gap=0.08, bucket_spread=0.02)
    write_target_quality(candidate, auc=0.62, pr_auc=0.56, brier=0.19, calibration_gap=0.07, bucket_spread=0.04)
    write_ml_summary(baseline, drawdown_auc=0.55, drawdown_brier=0.24)
    write_ml_summary(candidate, drawdown_auc=0.58, drawdown_brier=0.22)

    result = compare_research_runs(baseline, candidate)

    assert result["overall_status"] == "improved"
    assert result["recommendation"] == "commit_candidate"
    assert result["metric_deltas"]["outperformance_roc_auc"] == 0.020000000000000018
    assert result["metric_deltas"]["drawdown_risk_roc_auc"] == 0.029999999999999916


def test_compare_research_runs_returns_mixed_when_auc_gain_hurts_calibration(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_target_quality(baseline, auc=0.60, pr_auc=0.54, brier=0.22, calibration_gap=0.04, bucket_spread=0.02)
    write_target_quality(candidate, auc=0.62, pr_auc=0.55, brier=0.22, calibration_gap=0.09, bucket_spread=0.025)

    result = compare_research_runs(baseline, candidate)

    assert result["overall_status"] == "mixed"
    assert result["recommendation"] == "manual_review_required"
    assert "calibration gap worsened" in result["reason"]


def test_compare_research_runs_returns_worse_for_inverted_bucket_or_brier_decline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_target_quality(baseline, auc=0.60, pr_auc=0.54, brier=0.20, calibration_gap=0.05, bucket_spread=0.02)
    write_target_quality(candidate, auc=0.60, pr_auc=0.54, brier=0.24, calibration_gap=0.05, bucket_spread=-0.01)

    result = compare_research_runs(baseline, candidate)

    assert result["overall_status"] == "worse"
    assert result["recommendation"] == "do_not_commit_algorithm_change"
    assert "bucket_spread_inverted" in result["reason"]


def test_compare_research_runs_counts_regime_warnings_when_available(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_target_quality(baseline, auc=0.60, pr_auc=0.54, brier=0.20, calibration_gap=0.05, bucket_spread=0.02)
    write_target_quality(candidate, auc=0.61, pr_auc=0.55, brier=0.20, calibration_gap=0.05, bucket_spread=0.03)
    write_regime(baseline, ["Aligned"])
    write_regime(candidate, ["Aligned", "Inverted"])

    result = compare_research_runs(baseline, candidate)

    assert result["overall_status"] == "worse"
    assert "regime inversion count increased" in result["reason"]


def test_compare_research_runs_handles_missing_optional_csvs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_target_quality(baseline, auc=0.60, pr_auc=0.54, brier=0.22, calibration_gap=0.08, bucket_spread=0.02)
    write_target_quality(candidate, auc=0.61, pr_auc=0.55, brier=0.21, calibration_gap=0.08, bucket_spread=0.03)

    result = compare_research_runs(baseline, candidate)

    assert result["overall_status"] == "improved"
    assert result["warnings"] == []


def test_compare_cli_writes_json_output(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "candidate" / "iteration_comparison.json"
    write_target_quality(baseline, auc=0.60, pr_auc=0.54, brier=0.22, calibration_gap=0.08, bucket_spread=0.02)
    write_target_quality(candidate, auc=0.62, pr_auc=0.56, brier=0.20, calibration_gap=0.07, bucket_spread=0.04)

    exit_code = compare_runs_main(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--objective",
            "ml_target_research",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["overall_status"] == "improved"


def test_objective_templates_include_required_research_objectives() -> None:
    assert set(OBJECTIVE_TEMPLATES) == {
        "ml_target_research",
        "feature_engineering_research",
        "calibration_research",
        "regime_reliability_research",
    }


def test_codex_iteration_protocol_text_includes_required_stop_rules() -> None:
    text = Path("docs/codex_ml_iteration_protocol.md").read_text(encoding="utf-8")

    assert "Max iterations per Codex task: 2" in text
    assert "No production target switch unless explicitly instructed" in text
    assert "No `ml_score()` change unless explicitly instructed" in text
    assert "No generated research run outputs committed" in text
    assert "Compare baseline vs candidate export" in text
