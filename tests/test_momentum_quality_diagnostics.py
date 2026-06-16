from __future__ import annotations

import pandas as pd

from src.ml.diagnostics import build_momentum_quality_diagnostics
from src.research.export import export_research_lab_diagnostics


def price_path(kind: str, periods: int = 100) -> pd.Series:
    index = pd.RangeIndex(periods)
    if kind == "steady":
        return pd.Series([100.0 + day * 0.35 for day in index], dtype="float64")
    if kind == "gap":
        values = [100.0] * periods
        for day in range(60, periods):
            values[day] = 130.0
        return pd.Series(values, dtype="float64")
    if kind == "down":
        return pd.Series([130.0 - day * 0.25 for day in index], dtype="float64")
    return pd.Series([100.0] * periods, dtype="float64")


def baseline_rows(ticker: str, kind: str, regime: str = "Test regime") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    price = price_path(kind)
    previous_close = price.shift(1).fillna(price)
    if kind == "gap":
        open_price = previous_close.where(price.eq(previous_close), price)
        close = open_price.copy()
    else:
        open_price = previous_close
        close = price
    return pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ticker,
            "Open": open_price,
            "High": pd.concat([open_price, price], axis=1).max(axis=1) * 1.01,
            "Low": pd.concat([open_price, price], axis=1).min(axis=1) * 0.99,
            "Close": close,
            "Adj Close": price,
            "daily_return": price.pct_change(),
            "return_20d": price / price.shift(20) - 1.0,
            "return_60d": price / price.shift(60) - 1.0,
            "ma_50d": price.rolling(50, min_periods=50).mean(),
            "regime": regime,
        }
    )


def score_rows(
    baseline: pd.DataFrame,
    *,
    forward_excess_return: float,
    actual_out: int,
    actual_risk: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": baseline["Date"],
            "Ticker": baseline["Ticker"],
            "ML Score": 75.0 if actual_out else 35.0,
            "actual_out": actual_out,
            "forward_excess_return": forward_excess_return,
            "actual_risk": actual_risk,
        }
    )


def synthetic_panels(
    *,
    steady_return: float = 0.04,
    weak_return: float = -0.02,
    steady_risk: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    steady = baseline_rows("STEADY", "steady", regime="Regime A")
    gap = baseline_rows("GAP", "gap", regime="Regime B")
    down = baseline_rows("DOWN", "down", regime="Regime B")
    baseline = pd.concat([steady, gap, down], ignore_index=True)
    score = pd.concat(
        [
            score_rows(steady, forward_excess_return=steady_return, actual_out=int(steady_return > 0), actual_risk=steady_risk),
            score_rows(gap, forward_excess_return=weak_return, actual_out=int(weak_return > 0), actual_risk=0),
            score_rows(down, forward_excess_return=weak_return, actual_out=int(weak_return > 0), actual_risk=0),
        ],
        ignore_index=True,
    )
    return score, baseline


def test_momentum_quality_calculation_exports_expected_columns() -> None:
    score, baseline = synthetic_panels()

    diagnostics, by_regime, feature_summary = build_momentum_quality_diagnostics(
        score,
        baseline_panel=baseline,
        min_samples=10,
        min_bucket_size=3,
    )

    assert "momentum_quality_bucket" in diagnostics.columns
    assert {"ALL", "STEADY", "GAP", "DOWN"}.issubset(set(diagnostics["ticker"]))
    assert not by_regime.empty
    assert {"momentum_20d", "momentum_60d", "gap_adjusted_momentum_60d"}.issubset(
        set(feature_summary["feature"])
    )


def test_clean_momentum_scores_higher_than_gap_driven_momentum() -> None:
    score, baseline = synthetic_panels()

    diagnostics, _by_regime, _feature_summary = build_momentum_quality_diagnostics(
        score,
        baseline_panel=baseline,
        min_samples=10,
        min_bucket_size=3,
    )

    steady_high = diagnostics[
        (diagnostics["ticker"] == "STEADY") & (diagnostics["momentum_quality_bucket"] == "High")
    ]["sample_count"].sum()
    gap_high = diagnostics[
        (diagnostics["ticker"] == "GAP") & (diagnostics["momentum_quality_bucket"] == "High")
    ]["sample_count"].sum()
    assert steady_high > gap_high


def test_momentum_quality_missing_ohlcv_is_unavailable() -> None:
    score, baseline = synthetic_panels()

    diagnostics, by_regime, feature_summary = build_momentum_quality_diagnostics(
        score,
        baseline_panel=baseline.drop(columns=["Adj Close"]),
        min_samples=10,
        min_bucket_size=3,
    )

    assert diagnostics["classification"].tolist() == ["unavailable"]
    assert by_regime.empty
    assert feature_summary.empty


def test_momentum_quality_marks_insufficient_sample() -> None:
    score, baseline = synthetic_panels()

    diagnostics, _by_regime, _feature_summary = build_momentum_quality_diagnostics(
        score,
        baseline_panel=baseline,
        min_samples=500,
        min_bucket_size=100,
    )

    overall = diagnostics[diagnostics["ticker"] == "ALL"]
    assert set(overall["classification"]) == {"insufficient_sample"}


def test_momentum_quality_classifies_useful_mixed_and_harmful() -> None:
    useful_score, useful_baseline = synthetic_panels(steady_return=0.05, weak_return=-0.03)
    useful, _by_regime, _feature_summary = build_momentum_quality_diagnostics(
        useful_score,
        baseline_panel=useful_baseline,
        min_samples=10,
        min_bucket_size=3,
    )
    assert "useful" in set(useful[useful["ticker"] == "ALL"]["classification"])

    mixed_score, mixed_baseline = synthetic_panels(steady_return=0.01, weak_return=0.01)
    mixed, _by_regime, _feature_summary = build_momentum_quality_diagnostics(
        mixed_score,
        baseline_panel=mixed_baseline,
        min_samples=10,
        min_bucket_size=3,
    )
    assert set(mixed[mixed["ticker"] == "ALL"]["classification"]) == {"mixed"}

    harmful_score, harmful_baseline = synthetic_panels(steady_return=-0.04, weak_return=0.03)
    harmful, _by_regime, _feature_summary = build_momentum_quality_diagnostics(
        harmful_score,
        baseline_panel=harmful_baseline,
        min_samples=10,
        min_bucket_size=3,
    )
    assert "harmful" in set(harmful[harmful["ticker"] == "ALL"]["classification"])


def test_momentum_quality_export_writes_csv(tmp_path) -> None:
    score, baseline = synthetic_panels()
    diagnostics, _by_regime, _feature_summary = build_momentum_quality_diagnostics(
        score,
        baseline_panel=baseline,
        min_samples=10,
        min_bucket_size=3,
    )

    result = export_research_lab_diagnostics(
        run_metadata={"created_at": "2026-06-16T10:13:30", "ticker_count": 3},
        tables={"momentum_quality_diagnostics": diagnostics},
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "momentum_quality_diagnostics.csv")
    assert "classification" in exported.columns
    assert result.manifest["row_counts"]["momentum_quality_diagnostics.csv"] == len(diagnostics)
