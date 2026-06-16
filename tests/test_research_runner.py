from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.research.run_research_lab import parse_args, run_from_args


def test_headless_runner_argument_parser_accepts_expected_parameters(tmp_path: Path) -> None:
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\nMSFT,NVDA\n", encoding="utf-8")

    args = parse_args(
        [
            "--benchmark",
            "QQQ",
            "--feature-group",
            "all",
            "--model-mode",
            "auto_select",
            "--train-window",
            "504",
            "--test-window",
            "63",
            "--step",
            "63",
            "--embargo",
            "20",
            "--classification-threshold",
            "0.55",
            "--portfolio-name",
            "Lab",
            "--tickers",
            "TSLA,QQQ",
            "--tickers-file",
            str(tickers_file),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-name",
            "parser_run",
            "--export",
            "--quick",
        ]
    )

    assert args.benchmark == "QQQ"
    assert args.feature_group == "all"
    assert args.model_mode == "auto_select"
    assert args.train_window == 504
    assert args.classification_threshold == 0.55
    assert args.export is True
    assert args.quick is True


def test_headless_runner_exports_mocked_research_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_assemble(config):
        return {
            "run_metadata": {
                "benchmark": config.benchmark,
                "portfolio_name": config.portfolio_name,
                "ticker_count": len(config.tickers),
                "tickers": list(config.tickers),
                "feature_group": config.feature_group,
                "model_mode": config.model_mode,
                "train_window": config.train_window,
                "test_window": config.test_window,
                "step_size": config.step,
                "embargo_requested": config.embargo,
                "embargo_effective": 20,
                "classification_threshold": config.classification_threshold,
            },
            "tables": {
                "target_quality_summary": pd.DataFrame(
                    {
                        "target_id": ["outperform_20d"],
                        "overall_auc": [0.61],
                        "overall_bucket_spread": [0.03],
                        "production_candidate_status": ["Keep baseline"],
                        "recommended_next_step": ["keep production target"],
                    }
                ),
                "target_regime_comparison": pd.DataFrame(),
            },
        }

    monkeypatch.setattr("src.research.run_research_lab.assemble_research_lab_payload", fake_assemble)
    args = parse_args(
        [
            "--benchmark",
            "QQQ",
            "--tickers",
            "NVDA,MSFT",
            "--output-root",
            str(tmp_path / "data" / "research_runs"),
            "--run-name",
            "mock_run",
            "--export",
        ]
    )

    result = run_from_args(args)

    assert (result.run_dir / "run_metadata.json").exists()
    assert (result.run_dir / "diagnostics_manifest.json").exists()
    assert (result.run_dir / "codex_handoff.md").exists()
    assert (result.latest_dir / "run_metadata.json").exists()
    assert (result.latest_dir / "diagnostics_manifest.json").exists()
    assert (result.latest_dir / "codex_handoff.md").exists()
    assert result.manifest["tables_skipped"] == [{"table": "target_regime_comparison", "reason": "empty"}]


def test_headless_runner_module_does_not_require_streamlit() -> None:
    assert "streamlit" not in sys.modules
