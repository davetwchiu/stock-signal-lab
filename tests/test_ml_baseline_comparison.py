from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import build_ml_baseline_comparison


def score_panel(
    *,
    low_return: float = 0.00,
    medium_return: float = 0.03,
    high_return: float = 0.08,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    for index, date in enumerate(dates):
        if index < 10:
            ml_score_value = 20.0
            forward_return = low_return
        elif index < 20:
            ml_score_value = 55.0
            forward_return = medium_return
        else:
            ml_score_value = 85.0
            forward_return = high_return
        rows.append(
            {
                "Date": date,
                "Ticker": f"T{index:02d}",
                "ML Score": ml_score_value,
                "forward_return": forward_return,
                "forward_excess_return": forward_return,
            }
        )
    return pd.DataFrame(rows)


def baseline_panel(panel: pd.DataFrame, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": panel["Date"],
            "Ticker": panel["Ticker"],
            "return_60d": values,
        }
    )


def comparison_by_signal(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison.set_index("signal")


def test_ml_baseline_comparison_shows_ml_beats_no_skill_baseline() -> None:
    comparison = comparison_by_signal(build_ml_baseline_comparison(score_panel()))

    assert comparison.loc["ML score", "spread"] == pytest.approx(0.08)
    assert comparison.loc["No-skill / universe average", "spread"] == pytest.approx(0.0)
    assert "better bucket spread" in comparison.loc["ML score", "interpretation"]


def test_ml_baseline_comparison_adds_momentum_baseline_when_available() -> None:
    panel = score_panel()
    baseline = baseline_panel(panel, list(range(len(panel))))

    comparison = comparison_by_signal(build_ml_baseline_comparison(panel, baseline))

    assert "Momentum (60d)" in comparison.index
    assert comparison.loc["Momentum (60d)", "sample_size"] == len(panel)
    assert "baseline and ML score are similar" in comparison.loc["ML score", "interpretation"]


def test_ml_baseline_comparison_warns_when_simple_baseline_is_better() -> None:
    panel = score_panel(low_return=0.08, high_return=0.00)
    baseline = baseline_panel(panel, list(reversed(range(len(panel)))))

    comparison = comparison_by_signal(build_ml_baseline_comparison(panel, baseline))

    assert comparison.loc["Momentum (60d)", "spread"] > comparison.loc["ML score", "spread"]
    assert "does not improve" in comparison.loc["ML score", "interpretation"]
    assert "cautiously" in comparison.loc["ML score", "interpretation"]


def test_ml_baseline_comparison_handles_insufficient_sample() -> None:
    small = score_panel().head(4)

    comparison = comparison_by_signal(build_ml_baseline_comparison(small))

    assert pd.isna(comparison.loc["ML score", "spread"])
    assert comparison.loc["ML score", "direction"] == "insufficient"
    assert "too small" in comparison.loc["ML score", "interpretation"]


def test_ml_baseline_comparison_handles_missing_required_columns() -> None:
    missing_score = score_panel().drop(columns=["ML Score"])

    comparison = comparison_by_signal(build_ml_baseline_comparison(missing_score))

    assert pd.isna(comparison.loc["ML score", "spread"])
    assert comparison.loc["No-skill / universe average", "spread"] == pytest.approx(0.0)


def test_ml_baseline_comparison_handles_nan_heavy_input() -> None:
    panel = score_panel()
    panel.loc[:26, "forward_excess_return"] = pd.NA

    comparison = comparison_by_signal(build_ml_baseline_comparison(panel))

    assert pd.isna(comparison.loc["ML score", "spread"])
    assert comparison.loc["No-skill / universe average", "direction"] == "insufficient"


def test_ml_baseline_comparison_interpretation_avoids_trading_action_words() -> None:
    panel = score_panel()
    baseline = baseline_panel(panel, list(range(len(panel))))

    comparison = build_ml_baseline_comparison(panel, baseline)
    text = " ".join(comparison["interpretation"].astype(str)).lower()

    forbidden_patterns = [
        r"\bbuy\b",
        r"\bsell\b",
        r"\badd\b",
        r"\breduce\b",
        r"\bincrease position\b",
        r"\bdecrease position\b",
        r"\bchange allocation\b",
        r"\bchange ranking\b",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, text) is None
