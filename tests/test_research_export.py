from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.research.export import (
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
