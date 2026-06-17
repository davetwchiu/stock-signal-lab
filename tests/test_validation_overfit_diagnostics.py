from __future__ import annotations

import pandas as pd

from src.ml.diagnostics import (
    build_validation_fold_stability,
    build_validation_leakage_diagnostics,
    build_validation_overfit_warnings,
)
from src.research.export import build_codex_handoff


def fold_details(*, gap_days: int = 25, roc_values: tuple[float, ...] = (0.52, 0.54, 0.53)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, roc_auc in enumerate(roc_values, start=1):
        train_end = pd.Timestamp("2025-01-01") + pd.Timedelta(days=index * 90)
        test_start = train_end + pd.Timedelta(days=gap_days)
        rows.append(
            {
                "target": "outperformance",
                "fold": index,
                "train_start": train_end - pd.Timedelta(days=504),
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_start + pd.Timedelta(days=63),
                "roc_auc": roc_auc,
                "pr_auc": 0.48 + index * 0.01,
                "brier_score": 0.24 + index * 0.001,
                "effective_embargo": 20,
                "selected_model": "regularized_logistic",
            }
        )
    return pd.DataFrame(rows)


def test_validation_leakage_marks_clean_and_risky_gaps() -> None:
    clean = build_validation_leakage_diagnostics(fold_details(gap_days=30), label_horizon_days=20)
    risky = build_validation_leakage_diagnostics(fold_details(gap_days=5), label_horizon_days=20)

    assert clean.loc[0, "classification"] == "clean"
    assert clean.loc[0, "overlap_risk"] == "low"
    assert risky.loc[0, "classification"] == "risky"
    assert risky.loc[0, "overlap_risk"] == "high"


def test_validation_fold_stability_flags_one_lucky_fold() -> None:
    stable = build_validation_fold_stability(fold_details(roc_values=(0.52, 0.54, 0.53)))
    unstable = build_validation_fold_stability(fold_details(roc_values=(0.45, 0.72, 0.48)))

    assert stable.loc[0, "classification"] == "stable"
    assert unstable.loc[0, "classification"] == "unstable"
    assert unstable.loc[0, "fold_instability"] > 0.15


def test_validation_overfit_warnings_flag_ticker_concentration() -> None:
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    tickers = ["AAA"] * 40 + ["BBB"] * 10 + ["CCC"] * 10
    predictions = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": tickers,
            "actual": [1, 0] * 30,
            "probability": [0.2] * 20 + [0.5] * 20 + [0.8] * 20,
            "forward_excess_return": [0.01] * 20 + [0.02] * 20 + [0.03] * 20,
        }
    )
    baseline = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": tickers,
            "market_regime": ["up"] * 30 + ["down"] * 30,
        }
    )

    warnings = build_validation_overfit_warnings(
        predictions,
        baseline_panel=baseline,
        universe="AAA,BBB,CCC",
        min_samples=20,
        min_tickers=3,
        min_bucket_size=5,
    )
    indexed = warnings.set_index("diagnostic")

    assert indexed.loc["sample_coverage", "classification"] == "low_risk"
    assert indexed.loc["ticker_concentration", "classification"] == "likely_overfit"
    assert indexed.loc["regime_concentration", "classification"] == "low_risk"


def test_codex_handoff_includes_validation_diagnostics_summary() -> None:
    handoff = build_codex_handoff(
        run_metadata={
            "created_at": "2026-06-17T10:00:00",
            "benchmark": "QQQ",
            "feature_group": "all",
            "model_mode": "auto_select",
            "train_window": 504,
            "test_window": 63,
            "step_size": 63,
            "embargo_effective": 20,
            "ticker_count": 3,
        },
        tables={
            "validation_leakage_diagnostics": pd.DataFrame(
                {"diagnostic": ["train_test_gap"], "classification": ["clean"]}
            ),
            "validation_fold_stability": pd.DataFrame(
                {"target": ["outperformance"], "classification": ["stable"]}
            ),
            "validation_overfit_warnings": pd.DataFrame(
                {"diagnostic": ["ticker_concentration"], "classification": ["low_risk"]}
            ),
        },
        manifest={"files_written": []},
    )

    assert "## Validation leakage / overfit diagnostics" in handoff
    assert "Leakage rows=1: clean=1." in handoff
    assert "Fold-stability rows=1: stable=1." in handoff
    assert "Overfit-warning rows=1: low_risk=1." in handoff
