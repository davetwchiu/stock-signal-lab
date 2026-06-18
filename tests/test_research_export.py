from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.research.export import (
    build_opportunity_label_decision_summary,
    build_research_evidence_summary,
    build_research_lab_export_payload,
    export_research_lab_diagnostics,
    export_research_lab_payload,
    safe_filename_stem,
    zip_research_bundle,
)


def metadata() -> dict[str, object]:
    return {
        "created_at": "2026-06-16T10:13:30",
        "app_name": "Stock Signal Lab",
        "benchmark": "SPY",
        "portfolio_name": "Core",
        "ticker_count": 2,
        "tickers": ["AAA", "BBB"],
        "feature_group": "all",
        "model_mode": "auto_select",
        "train_window": 504,
        "test_window": 63,
        "step_size": 63,
        "embargo_requested": 20,
        "embargo_effective": 20,
        "classification_threshold": 0.5,
        "target_candidates_enabled": True,
        "extended_target_comparison_enabled": True,
        "data_start": "2023-06-16",
        "data_end": "2026-06-16",
        "git_commit": "abc1234",
    }


def test_export_writes_required_files(tmp_path: Path) -> None:
    result = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={},
        output_root=tmp_path,
        run_id="2026-06-16_101330",
    )

    assert (result.run_dir / "run_metadata.json").exists()
    assert (result.run_dir / "diagnostics_manifest.json").exists()
    assert (result.run_dir / "codex_handoff.md").exists()


def test_export_writes_non_empty_dataframes_to_csv(tmp_path: Path) -> None:
    table = pd.DataFrame({"target_id": ["outperform_20d"], "roc_auc": [0.61]})

    result = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={"target_walk_forward_comparison": table},
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "target_walk_forward_comparison.csv")
    assert exported.to_dict("records") == [{"target_id": "outperform_20d", "roc_auc": 0.61}]
    assert result.manifest["row_counts"]["target_walk_forward_comparison.csv"] == 1
    assert result.manifest["column_names"]["target_walk_forward_comparison.csv"] == [
        "target_id",
        "roc_auc",
    ]


def test_empty_and_missing_optional_tables_are_skipped(tmp_path: Path) -> None:
    result = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={
            "ml_diagnostics_summary": pd.DataFrame(),
            "target_quality_summary": None,
            "notes": {"not": "a dataframe"},
        },
        output_root=tmp_path,
        run_id="run",
    )

    assert not (result.run_dir / "ml_diagnostics_summary.csv").exists()
    assert result.manifest["tables_skipped"] == [
        {"table": "ml_diagnostics_summary", "reason": "empty"},
        {"table": "target_quality_summary", "reason": "missing"},
        {"table": "notes", "reason": "not_dataframe"},
    ]


def test_latest_is_updated_deterministically(tmp_path: Path) -> None:
    first = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={"ml_diagnostics_summary": pd.DataFrame({"metric": ["old"], "value": [1]})},
        output_root=tmp_path,
        run_id="first",
    )
    second = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={"ml_diagnostics_summary": pd.DataFrame({"metric": ["new"], "value": [2]})},
        output_root=tmp_path,
        run_id="second",
    )

    assert first.latest_dir == second.latest_dir
    latest = pd.read_csv(tmp_path / "latest" / "ml_diagnostics_summary.csv")
    assert latest.to_dict("records") == [{"metric": "new", "value": 2}]
    assert not (tmp_path / "latest" / "stale.csv").exists()


def test_codex_handoff_contains_required_sections(tmp_path: Path) -> None:
    result = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={
            "target_quality_summary": pd.DataFrame(
                {
                    "target_id": ["outperform_20d", "tail_adjusted_outperform_20d"],
                    "overall_target_quality": ["Mixed", "Promising"],
                    "production_candidate_status": ["Keep baseline", "Trial candidate"],
                    "recommended_next_step": ["keep production target", "research reliability preview"],
                    "best_feature_group": ["technical", "all"],
                    "regime_stability": ["Mixed", "Stable"],
                    "calibration_quality": ["Weak", "Good"],
                    "candidate_rank": [2, 1],
                }
            ),
            "ml_reliability_by_regime": pd.DataFrame(
                {
                    "regime": ["Uptrend", "Downtrend"],
                    "classification": ["mixed", "inverted"],
                }
            ),
            "ml_reliability_gate_diagnostics": pd.DataFrame(
                {
                    "gate_name": ["no_gate_baseline", "exclude_inverted_regimes"],
                    "classification": ["mixed", "harmful"],
                }
            ),
        },
        output_root=tmp_path,
        run_id="run",
    )

    handoff = result.codex_handoff
    assert "## Run metadata" in handoff
    assert "## Main evidence" in handoff
    assert "## Production target status" in handoff
    assert "## ML reliability by regime" in handoff
    assert (
        "This table shows where ML score historically worked, failed, or lacked enough evidence."
        in handoff
    )
    assert "## ML reliability gate diagnostics" in handoff
    assert "Research-only reliability gates were tested without changing production scoring." in handoff
    assert "## Suggested next engineering direction" in handoff
    assert "## How Codex should use this bundle" in handoff
    assert "no production target switch is supported" in handoff
    assert "## Codex instructions for next iteration" in handoff


def test_research_evidence_summary_plainly_surfaces_negative_and_missing_evidence() -> None:
    summary = build_research_evidence_summary(
        {
            "ml_reliability_by_regime": pd.DataFrame({"classification": ["mixed", "inverted"]}),
            "ml_reliability_gate_diagnostics": pd.DataFrame({"classification": ["harmful"]}),
            "momentum_quality_diagnostics": pd.DataFrame({"classification": ["useful", "harmful"]}),
            "earnings_pead_summary": pd.DataFrame(
                {
                    "classification": ["unavailable"],
                    "pead_signal_direction": ["unavailable"],
                    "ml_near_earnings_effect": ["unavailable"],
                }
            ),
            "validation_leakage_diagnostics": pd.DataFrame({"classification": ["watch"]}),
            "validation_fold_stability": pd.DataFrame({"classification": ["mixed"]}),
            "validation_overfit_warnings": pd.DataFrame({"classification": ["low_risk"]}),
            "portfolio_crowding_summary": pd.DataFrame(
                {"classification": ["moderate_crowding"], "high_overlap_pair_count": [2], "largest_cluster_size": [3]}
            ),
            "portfolio_factor_crowding_summary": pd.DataFrame({"classification": ["crowded"]}),
            "feature_importance_production_readiness": pd.DataFrame(
                {"classification": ["watch"], "reason": ["Some stable features exist."]}
            ),
        }
    )

    indexed = summary.set_index("area")
    assert list(summary.columns) == [
        "area",
        "latest_classification",
        "evidence_strength",
        "production_readiness",
        "key_reason",
        "recommended_next_action",
    ]
    assert indexed.loc["ML reliability by regime", "latest_classification"] == "inverted"
    assert indexed.loc["Momentum quality", "evidence_strength"] == "harmful"
    assert indexed.loc["Earnings / PEAD", "production_readiness"] == "unavailable"
    assert indexed.loc["Portfolio crowding", "latest_classification"] == "crowded"
    assert indexed.loc["Overall production readiness", "production_readiness"] == "not_ready"
    assert "No current diagnostic gives stable enough evidence" in indexed.loc[
        "Overall production readiness", "key_reason"
    ]


def test_research_evidence_summary_handles_missing_diagnostics() -> None:
    summary = build_research_evidence_summary({})

    indexed = summary.set_index("area")
    assert indexed.loc["ML reliability gate", "latest_classification"] == "unavailable"
    assert indexed.loc["Feature importance stability", "production_readiness"] == "unavailable"
    assert indexed.loc["Overall production readiness", "production_readiness"] == "not_ready"


def test_opportunity_label_decision_summary_keeps_fragile_risk_adjusted_audit_only() -> None:
    summary = build_opportunity_label_decision_summary(
        {
            "target_stop_rule_comparison": pd.DataFrame(
                {
                    "target_id": ["outperform_20d", "risk_adjusted_excess_20d"],
                    "bucket_spread": [0.23, 0.24],
                    "worst_ticker": ["AVGO", "NVDA"],
                    "worst_ticker_bucket_spread": [-0.26, -0.38],
                    "worst_regime": ["Uptrend / high volatility", "Downtrend / high risk"],
                    "worst_regime_bucket_spread": [-0.07, -0.03],
                    "regime_inversion_count": [1, 0],
                    "recommended_decision": ["Continue", "Continue"],
                    "stop_rule_result": ["baseline", "pass"],
                    "failure_diagnosis": [
                        "Current production target baseline for comparison.",
                        "Candidate clears the research stop rule versus the current baseline.",
                    ],
                    "production_change_justified": [False, False],
                }
            ),
            "target_quality_summary": pd.DataFrame(
                {
                    "target_id": ["outperform_20d", "risk_adjusted_excess_20d"],
                    "display_name": ["Current 20d outperformance", "Recent-vol adjusted excess"],
                    "overall_target_quality": ["Mixed", "Mixed"],
                    "production_candidate_status": ["Keep baseline", "Research-only candidate"],
                    "regime_stability": ["Inverted in some regimes", "Regime-sensitive"],
                }
            ),
            "target_arena_comparison": pd.DataFrame(
                {
                    "target_id": ["outperform_20d", "risk_adjusted_excess_20d"],
                    "evidence_classification": ["mixed", "promising"],
                }
            ),
            "risk_adjusted_opportunity_fragility": pd.DataFrame(
                {
                    "view": ["high_vol_uptrend_exclude_pltr_tsla"],
                    "excluded_tickers": ["PLTR,TSLA"],
                    "model_bucket_spread": [0.19],
                }
            ),
        }
    )

    indexed = summary.set_index("target_name")
    risk_adjusted = indexed.loc["Recent-vol adjusted excess"]
    assert risk_adjusted["recommended_decision"] == "Hold as audit-only"
    assert risk_adjusted["weakest_tickers"] == "PLTR,TSLA"
    assert risk_adjusted["weakest_regime"] == "Uptrend / high volatility"
    assert risk_adjusted["inversion_hypothesis_result"] == "ticker-driven; do not invert regime-wide"
    assert bool(risk_adjusted["audit_only_flag"]) is True


def test_opportunity_label_decision_summary_missing_fields_needs_more_data() -> None:
    summary = build_opportunity_label_decision_summary(
        {
            "target_stop_rule_comparison": pd.DataFrame(
                {
                    "target_id": ["risk_adjusted_excess_20d"],
                    "recommended_decision": ["Continue"],
                }
            )
        }
    )

    row = summary.iloc[0]
    assert row["recommended_decision"] == "Needs more data"
    assert "Missing source fields" in row["short_reason"]


def test_export_includes_opportunity_label_decision_summary_csv(tmp_path: Path) -> None:
    result = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={
            "target_stop_rule_comparison": pd.DataFrame(
                {
                    "target_id": ["risk_adjusted_excess_20d"],
                    "bucket_spread": [0.24],
                    "worst_ticker": ["NVDA"],
                    "worst_ticker_bucket_spread": [-0.38],
                    "worst_regime": ["Downtrend / high risk"],
                    "worst_regime_bucket_spread": [-0.03],
                    "regime_inversion_count": [0],
                    "recommended_decision": ["Continue"],
                    "stop_rule_result": ["pass"],
                    "failure_diagnosis": ["Candidate clears the research stop rule versus the current baseline."],
                    "production_change_justified": [False],
                }
            ),
            "target_quality_summary": pd.DataFrame(
                {
                    "target_id": ["risk_adjusted_excess_20d"],
                    "display_name": ["Recent-vol adjusted excess"],
                    "overall_target_quality": ["Mixed"],
                    "production_candidate_status": ["Research-only candidate"],
                    "regime_stability": ["Regime-sensitive"],
                }
            ),
        },
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "opportunity_label_decision_summary.csv")
    assert "opportunity_label_decision_summary.csv" in result.manifest["files_written"]
    assert result.manifest["row_counts"]["opportunity_label_decision_summary.csv"] == 1
    assert exported.loc[0, "recommended_decision"] == "Continue"


def test_export_includes_research_evidence_summary_csv(tmp_path: Path) -> None:
    result = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={"ml_reliability_gate_diagnostics": pd.DataFrame({"classification": ["harmful"]})},
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "research_evidence_summary.csv")
    assert "research_evidence_summary.csv" in result.manifest["files_written"]
    assert result.manifest["row_counts"]["research_evidence_summary.csv"] == len(exported)
    assert exported.loc[exported["area"].eq("ML reliability gate"), "evidence_strength"].iloc[0] == "harmful"


def test_app_facing_payload_adds_research_only_summary_without_production_inputs() -> None:
    payload = build_research_lab_export_payload(
        run_metadata=metadata(),
        tables={"ml_reliability_gate_diagnostics": pd.DataFrame({"classification": ["mixed"]})},
    )

    summary = payload["tables"]["research_evidence_summary"]
    assert isinstance(summary, pd.DataFrame)
    assert summary.loc[summary["area"].eq("Overall production readiness"), "production_readiness"].iloc[0] == "not_ready"
    assert "scoring, action, sizing, ranking, or allocation changes" in summary.loc[
        summary["area"].eq("Overall production readiness"), "key_reason"
    ].iloc[0]


def test_export_does_not_mutate_input_dataframes(tmp_path: Path) -> None:
    frame = pd.DataFrame({"target id": ["AAA"], "score": [1.0]})
    before = frame.copy(deep=True)

    export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={"unsafe display name": frame},
        output_root=tmp_path,
        run_id="run",
    )

    pd.testing.assert_frame_equal(frame, before)


def test_filenames_are_stable_and_safe(tmp_path: Path) -> None:
    assert safe_filename_stem("ML Diagnostics Summary") == "ml_diagnostics_summary"
    assert safe_filename_stem("target-quality.summary") == "target_quality_summary"

    export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={"ML Diagnostics Summary": pd.DataFrame({"x": [1]})},
        output_root=tmp_path,
        run_id="run",
    )

    assert (tmp_path / "run" / "ml_diagnostics_summary.csv").exists()


def test_generated_output_folder_is_local_research_run(tmp_path: Path) -> None:
    result = export_research_lab_diagnostics(
        run_metadata=metadata(),
        tables={},
        output_root=tmp_path / "data" / "research_runs",
        run_id="run",
    )

    assert result.run_dir == tmp_path / "data" / "research_runs" / "run"
    assert "research_runs" in str(result.run_dir)


def test_export_accepts_datetime_created_at_and_zip_is_dependency_free(tmp_path: Path) -> None:
    result = export_research_lab_diagnostics(
        run_metadata={**metadata(), "created_at": datetime(2026, 6, 16, 10, 13, 30)},
        tables={"ml_score_buckets": pd.DataFrame({"bucket": ["High"], "count": [3]})},
        output_root=tmp_path,
        run_id="run",
    )

    archive = zip_research_bundle(result.run_dir)
    assert b"ml_score_buckets.csv" in archive
    saved_metadata = json.loads((result.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["created_at"] == "2026-06-16T10:13:30"


def test_app_facing_payload_export_writes_repo_relative_research_bundle(tmp_path: Path) -> None:
    payload = build_research_lab_export_payload(
        run_metadata=metadata(),
        tables={
            "ml_diagnostics_summary": pd.DataFrame(
                {"target": ["outperformance"], "roc_auc": [0.62]}
            )
        },
    )
    output_root = tmp_path / "data" / "research_runs"

    result = export_research_lab_payload(
        payload,
        output_root=output_root,
        run_id="2026-06-16_101330",
    )

    expected_run_dir = tmp_path / "data" / "research_runs" / "2026-06-16_101330"
    latest_dir = tmp_path / "data" / "research_runs" / "latest"
    assert result.run_dir == expected_run_dir
    assert result.latest_dir == latest_dir
    for filename in ("run_metadata.json", "diagnostics_manifest.json", "codex_handoff.md"):
        assert (expected_run_dir / filename).exists()
        assert (latest_dir / filename).exists()
    assert (expected_run_dir / "ml_diagnostics_summary.csv").exists()
