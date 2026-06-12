from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import (
    build_feature_family_signal_summary,
    build_feature_quantile_signal_summary,
    build_feature_signal_table,
    build_feature_signal_warnings,
    build_ml_feature_signal_diagnostics,
)


FORBIDDEN_PATTERNS = [
    r"\bbuy\b",
    r"\bsell\b",
    r"\badd\b",
    r"\breduce\b",
    r"\bincrease position\b",
    r"\bdecrease position\b",
    r"\bchange allocation\b",
    r"\bchange ranking\b",
]


def signal_panel(rows: int = 60) -> pd.DataFrame:
    index = range(rows)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Ticker": ["AAA" if row < rows / 2 else "BBB" for row in index],
            "return_20d": [row / 100 for row in index],
            "dist_ma_50d": [row / 300 for row in index],
            "volatility_20d": [0.3 - row / 400 for row in index],
            "rs_spy_60d": [0.05 + row / 500 for row in index],
            "volume_z_20d": [row % 7 for row in index],
            "rsi_14d": [30 + row % 25 for row in index],
            "fourier_1": [row / 50 for row in index],
            "wavelet_1": [row / 55 for row in index],
            "constant_feature": 1.0,
            "forward_20d_excess_return": [row / 1000 for row in index],
            "forward_20d_return": [0.02 - row / 2000 for row in index],
            "label_outperform_20d": [1 if row >= rows / 2 else 0 for row in index],
            "label_drawdown_risk_20d": [1 if row < rows / 3 else 0 for row in index],
        }
    )


def test_feature_signal_table_scores_existing_targets_and_labels() -> None:
    table = build_feature_signal_table(
        signal_panel(),
        ["return_20d", "volatility_20d", "constant_feature"],
    )

    row = table[table["feature"] == "return_20d"].iloc[0]

    assert row["family"] == "momentum / return"
    assert row["valid_sample_count"] == 60
    assert row["missing_rate"] == pytest.approx(0.0)
    assert row["unique_value_count"] == 60
    assert row["spearman_to_return_target"] == pytest.approx(1.0)
    assert row["abs_spearman_to_return_label"] > 0.80
    assert row["abs_spearman_to_drawdown_label"] > 0.70
    assert row["top_quantile_target_mean"] > row["bottom_quantile_target_mean"]
    assert row["quantile_spread"] > 0


def test_feature_signal_table_falls_back_to_forward_return_target() -> None:
    data = signal_panel().drop(columns=["forward_20d_excess_return"])

    row = build_feature_signal_table(data, ["return_20d"]).iloc[0]

    assert row["spearman_to_return_target"] == pytest.approx(-1.0)


def test_feature_family_signal_summary_groups_top_features() -> None:
    table = build_feature_signal_table(
        signal_panel(),
        ["return_20d", "dist_ma_50d", "volatility_20d", "volume_z_20d"],
    )
    summary = build_feature_family_signal_summary(table)

    assert set(summary["family"]) >= {
        "momentum / return",
        "trend / moving average",
        "volatility",
        "volume / liquidity",
    }
    top_row = summary.iloc[0]
    assert top_row["max_abs_signal"] >= 0.90
    assert top_row["top_feature_signal"] >= 0.90
    assert top_row["share_of_top_features"] > 0


def test_feature_quantile_signal_summary_limits_to_top_spreads() -> None:
    summary = build_feature_quantile_signal_summary(
        signal_panel(),
        ["return_20d", "volatility_20d", "volume_z_20d"],
        max_features=2,
    )

    assert len(summary) == 2
    assert summary["abs_quantile_spread"].is_monotonic_decreasing
    assert set(summary["feature"]).issubset({"return_20d", "volatility_20d", "volume_z_20d"})


def test_feature_signal_warnings_flag_complex_concentration_and_redundancy() -> None:
    data = signal_panel()
    table = build_feature_signal_table(data, ["fourier_1", "wavelet_1", "return_20d"])
    family_summary = build_feature_family_signal_summary(table)
    high_correlation_pairs = pd.DataFrame(
        [
            {
                "feature_1": "fourier_1",
                "feature_2": "wavelet_1",
                "abs_correlation": 1.0,
                "interpretation": "These features are highly correlated in this sample.",
            }
        ]
    )

    warnings = build_feature_signal_warnings(
        data,
        table,
        family_summary,
        high_correlation_pairs,
    )

    assert "complex feature signal concentration" in set(warnings["warning"])
    assert "top signal features are redundant" in set(warnings["warning"])


def test_feature_signal_diagnostics_handles_missing_targets_and_constant_features() -> None:
    data = signal_panel().drop(
        columns=[
            "forward_20d_excess_return",
            "forward_20d_return",
            "label_outperform_20d",
            "label_drawdown_risk_20d",
        ]
    )

    diagnostics = build_ml_feature_signal_diagnostics(data, ["constant_feature"])

    assert diagnostics.signal_table.loc[0, "unique_value_count"] == 1
    assert "no suitable target or label" in set(diagnostics.warnings["warning"])
    assert "too few unique values" in set(diagnostics.warnings["warning"])


def test_feature_signal_diagnostics_handles_empty_input() -> None:
    diagnostics = build_ml_feature_signal_diagnostics(pd.DataFrame(), ["return_20d"])

    assert diagnostics.signal_table.empty
    assert diagnostics.family_summary.empty
    assert diagnostics.quantile_summary.empty
    assert diagnostics.warnings.loc[0, "warning"] == "feature signal unavailable"


def test_feature_signal_interpretations_avoid_trading_action_words() -> None:
    diagnostics = build_ml_feature_signal_diagnostics(
        signal_panel(),
        ["return_20d", "fourier_1", "wavelet_1", "constant_feature"],
        high_correlation_pairs=pd.DataFrame(
            [
                {
                    "feature_1": "fourier_1",
                    "feature_2": "wavelet_1",
                    "abs_correlation": 1.0,
                    "interpretation": "These features are highly correlated in this sample.",
                }
            ]
        ),
    )
    tables = [
        diagnostics.signal_table,
        diagnostics.family_summary,
        diagnostics.quantile_summary,
        diagnostics.warnings,
    ]
    text = " ".join(
        str(value)
        for table in tables
        for value in table.get("interpretation", pd.Series(dtype=str)).dropna()
    ).lower()

    for pattern in FORBIDDEN_PATTERNS:
        assert re.search(pattern, text) is None
