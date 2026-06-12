from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import build_regime_segmented_ml_diagnostics


def score_panel(
    regimes: dict[str, tuple[float, float, float]] | None = None,
    *,
    rows_per_bucket: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_regimes = regimes or {
        "Uptrend / low volatility": (0.00, 0.03, 0.08),
        "Downtrend / high risk": (0.04, 0.03, 0.04),
    }
    rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    date_index = pd.date_range("2024-01-01", periods=len(active_regimes) * rows_per_bucket * 3, freq="B")
    row_index = 0
    for regime, (low_return, medium_return, high_return) in active_regimes.items():
        for bucket, ml_score_value, forward_return in (
            ("low", 20.0, low_return),
            ("medium", 55.0, medium_return),
            ("high", 85.0, high_return),
        ):
            for bucket_index in range(rows_per_bucket):
                date = date_index[row_index]
                ticker = f"{regime[:2]}{bucket[:1]}{bucket_index:02d}"
                rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "ML Score": ml_score_value,
                        "forward_return": forward_return,
                    }
                )
                baseline_rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "regime": regime,
                    }
                )
                row_index += 1
    return pd.DataFrame(rows), pd.DataFrame(baseline_rows)


def diagnostics_by_regime(diagnostics: pd.DataFrame) -> pd.DataFrame:
    return diagnostics.set_index("regime")


def test_regime_segmented_diagnostics_show_regime_dependent_separation() -> None:
    panel, baseline = score_panel()

    diagnostics = diagnostics_by_regime(
        build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline)
    )

    assert diagnostics.loc["Uptrend / low volatility", "spread"] == pytest.approx(0.08)
    assert diagnostics.loc["Uptrend / low volatility", "direction"] == "top bucket higher"
    assert "positive separation" in diagnostics.loc["Uptrend / low volatility", "interpretation"]
    assert diagnostics.loc["Downtrend / high risk", "evidence_quality"] == "mixed"
    assert "weak or inconclusive" in diagnostics.loc["Downtrend / high risk", "interpretation"]


def test_regime_segmented_diagnostics_can_show_positive_separation_in_both_regimes() -> None:
    panel, baseline = score_panel(
        {
            "Uptrend / low volatility": (0.00, 0.02, 0.07),
            "Sideways / mixed": (-0.02, 0.01, 0.04),
        }
    )

    diagnostics = build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline)

    assert set(diagnostics["evidence_quality"]) == {"usable"}
    assert all(diagnostics["direction"] == "top bucket higher")


def test_regime_segmented_diagnostics_warns_on_inverted_regime() -> None:
    panel, baseline = score_panel({"Distribution": (0.06, 0.02, -0.01)})

    diagnostics = diagnostics_by_regime(
        build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline)
    )

    assert diagnostics.loc["Distribution", "spread"] < 0
    assert diagnostics.loc["Distribution", "direction"] == "bottom bucket higher"
    assert diagnostics.loc["Distribution", "evidence_quality"] == "inverted"
    assert "inverted" in diagnostics.loc["Distribution", "interpretation"]


def test_regime_segmented_diagnostics_handles_insufficient_regime_sample() -> None:
    panel, baseline = score_panel({"Uptrend / low volatility": (0.00, 0.02, 0.07)}, rows_per_bucket=2)

    diagnostics = diagnostics_by_regime(
        build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline)
    )

    assert pd.isna(diagnostics.loc["Uptrend / low volatility", "spread"])
    assert diagnostics.loc["Uptrend / low volatility", "direction"] == "insufficient"
    assert "too small" in diagnostics.loc["Uptrend / low volatility", "interpretation"]


def test_regime_segmented_diagnostics_handles_missing_regime_column() -> None:
    panel, baseline = score_panel()

    diagnostics = build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline.drop(columns=["regime"]))

    assert diagnostics.empty


def test_regime_segmented_diagnostics_handles_missing_required_columns() -> None:
    panel, baseline = score_panel()

    missing_score = build_regime_segmented_ml_diagnostics(panel.drop(columns=["ML Score"]), baseline_panel=baseline)
    missing_forward_return = build_regime_segmented_ml_diagnostics(
        panel.drop(columns=["forward_return"]),
        baseline_panel=baseline,
    )

    assert missing_score.empty
    assert missing_forward_return.empty


def test_regime_segmented_diagnostics_handles_nan_heavy_input() -> None:
    panel, baseline = score_panel()
    panel.loc[:42, "forward_return"] = pd.NA

    diagnostics = build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline)

    assert diagnostics.empty or diagnostics["direction"].eq("insufficient").all()


def test_regime_segmented_diagnostics_flags_one_regime_only_input() -> None:
    panel, baseline = score_panel({"Uptrend / low volatility": (0.00, 0.02, 0.07)})

    diagnostics = diagnostics_by_regime(
        build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline)
    )

    assert "concentrated in one regime" in diagnostics.loc["Uptrend / low volatility", "interpretation"]


def test_regime_segmented_diagnostics_interpretation_avoids_trading_action_words() -> None:
    panel, baseline = score_panel()

    diagnostics = build_regime_segmented_ml_diagnostics(panel, baseline_panel=baseline)
    text = " ".join(diagnostics["interpretation"].astype(str)).lower()

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
