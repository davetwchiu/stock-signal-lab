from __future__ import annotations

import pandas as pd

from src.ml.labels import make_forward_labels
from src.research.export import export_research_lab_diagnostics
from src.research.lab import (
    STRESS_RELATIVE_STRENGTH_COLUMNS,
    STRESS_RELATIVE_STRENGTH_SNAPSHOT_COLUMNS,
    build_stress_relative_strength_diagnostics,
    latest_stress_relative_strength_snapshot,
)


def synthetic_panel() -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    benchmark_returns = pd.Series(0.001, index=dates)
    benchmark_returns.iloc[[20, 21, 55, 56, 85]] = [-0.02, -0.018, -0.021, -0.017, -0.02]
    benchmark_returns.iloc[[22, 57, 86]] = [0.016, 0.014, 0.018]
    benchmark = 100.0 * (1.0 + benchmark_returns).cumprod()

    frames = []
    for ticker, stress_edge, rebound_edge in (
        ("STRONG", 0.012, 0.012),
        ("MID", 0.002, 0.002),
        ("WEAK", -0.012, -0.008),
    ):
        returns = benchmark_returns.copy()
        stress = benchmark_returns <= -0.015
        rebound = benchmark_returns >= 0.01
        returns.loc[stress] = benchmark_returns.loc[stress] + stress_edge
        returns.loc[rebound] = benchmark_returns.loc[rebound] + rebound_edge
        price = 100.0 * (1.0 + returns).cumprod()
        frame = pd.DataFrame(
            {
                "Ticker": ticker,
                "Date": dates,
                "Open": price.shift(1).fillna(price.iloc[0]),
                "High": price * 1.01,
                "Low": price * 0.99,
                "Close": price,
                "Adj Close": price,
                "Volume": 1_000_000,
                "daily_return": returns,
                "rs_qqq_60d": price.pct_change(60) - benchmark.pct_change(60),
                "dist_ma_200d": 0.01,
            }
        )
        labels = make_forward_labels(frame.set_index("Date"), benchmark_price=benchmark, horizon=20)
        frames.append(frame.join(labels, on="Date"))
    return pd.concat(frames, ignore_index=True), benchmark


def test_stress_day_detection_counts_benchmark_drawdowns() -> None:
    panel, benchmark = synthetic_panel()

    diagnostics = build_stress_relative_strength_diagnostics(
        panel,
        benchmark,
        stress_window=10,
        min_stress_days=1,
    )

    row = diagnostics[(diagnostics["ticker"] == "STRONG") & (diagnostics["date"] == "2024-01-30")].iloc[0]
    assert row["stress_day_count"] == 2
    assert row["stress_excess_return"] > 0


def test_rebound_leadership_does_not_use_future_rebound_days() -> None:
    panel, benchmark = synthetic_panel()

    diagnostics = build_stress_relative_strength_diagnostics(
        panel,
        benchmark,
        stress_window=10,
        min_stress_days=1,
    )

    before_rebound = diagnostics[
        (diagnostics["ticker"] == "STRONG") & (diagnostics["date"] == "2024-01-30")
    ].iloc[0]
    after_rebound = diagnostics[
        (diagnostics["ticker"] == "STRONG") & (diagnostics["date"] == "2024-01-31")
    ].iloc[0]
    assert before_rebound["rebound_day_count"] == 0
    assert after_rebound["rebound_day_count"] == 1


def test_resilient_ticker_ranks_above_weak_ticker_during_stress() -> None:
    panel, benchmark = synthetic_panel()

    diagnostics = build_stress_relative_strength_diagnostics(
        panel,
        benchmark,
        stress_window=40,
        min_stress_days=1,
    )

    latest = diagnostics[diagnostics["date"] == "2024-05-01"].set_index("ticker")
    assert latest.loc["STRONG", "stress_relative_strength_score"] > latest.loc["WEAK", "stress_relative_strength_score"]
    assert latest.loc["STRONG", "stress_rs_bucket"] == "High"


def test_latest_stress_relative_strength_snapshot_is_sorted_and_sample_gated() -> None:
    panel, benchmark = synthetic_panel()
    diagnostics = build_stress_relative_strength_diagnostics(
        panel,
        benchmark,
        stress_window=40,
        min_stress_days=1,
    )

    snapshot = latest_stress_relative_strength_snapshot(diagnostics, min_stress_days=1)

    assert list(snapshot.columns) == STRESS_RELATIVE_STRENGTH_SNAPSHOT_COLUMNS
    assert snapshot["ticker"].tolist() == ["STRONG", "MID", "WEAK"]
    assert snapshot["sample_status"].eq("sufficient").all()
    assert snapshot["date"].nunique() == 1


def test_latest_stress_relative_strength_snapshot_handles_missing_data() -> None:
    snapshot = latest_stress_relative_strength_snapshot(pd.DataFrame({"ticker": ["NVDA"]}))

    assert snapshot.empty
    assert list(snapshot.columns) == STRESS_RELATIVE_STRENGTH_SNAPSHOT_COLUMNS


def test_stress_relative_strength_export_shape(tmp_path) -> None:
    panel, benchmark = synthetic_panel()
    diagnostics = build_stress_relative_strength_diagnostics(
        panel,
        benchmark,
        stress_window=40,
        min_stress_days=1,
    )
    snapshot = latest_stress_relative_strength_snapshot(diagnostics, min_stress_days=1)

    result = export_research_lab_diagnostics(
        run_metadata={"created_at": "2026-07-09T10:00:00", "benchmark": "QQQ", "ticker_count": 3},
        tables={
            "stress_relative_strength_diagnostics": diagnostics,
            "stress_relative_strength_snapshot": snapshot,
        },
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "stress_relative_strength_diagnostics.csv")
    assert list(exported.columns) == STRESS_RELATIVE_STRENGTH_COLUMNS
    assert result.manifest["row_counts"]["stress_relative_strength_diagnostics.csv"] == len(diagnostics)
    assert result.manifest["row_counts"]["stress_relative_strength_snapshot.csv"] == 3
