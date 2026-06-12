from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import (
    build_drawdown_label_threshold_sensitivity,
    build_label_distribution,
    build_label_prevalence_summary,
    build_ml_label_audit,
    build_return_drawdown_label_overlap,
    build_return_label_threshold_sensitivity,
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


def panel(rows: int = 30) -> pd.DataFrame:
    index = range(rows)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Ticker": ["AAA" if row < rows / 2 else "BBB" for row in index],
            "regime": ["Uptrend / low volatility" if row % 2 == 0 else "Downtrend / high risk" for row in index],
            "label_outperform_20d": [1 if row % 2 == 0 else 0 for row in index],
            "label_drawdown_risk_20d": [1 if row < rows / 5 else 0 for row in index],
            "forward_20d_excess_return": [-0.02 if row < 10 else 0.015 if row < 20 else 0.06 for row in index],
            "forward_20d_drawdown": [-0.03 if row < 10 else -0.12 if row < 20 else -0.25 for row in index],
        }
    )


def first_row(frame: pd.DataFrame) -> pd.Series:
    assert len(frame) == 1
    return frame.iloc[0]


def test_label_prevalence_identifies_balanced_label() -> None:
    summary = build_label_prevalence_summary(panel(), label_columns=["label_outperform_20d"])

    row = first_row(summary)

    assert row["sample_size"] == 30
    assert row["positive_count"] == 15
    assert row["positive_rate"] == pytest.approx(0.5)
    assert row["class_balance"] == "balanced"
    assert row["interpretation"] == "This label has a balanced positive rate in this sample."


def test_label_prevalence_identifies_sparse_positive_label() -> None:
    data = panel()
    data["label_outperform_20d"] = [1 if row < 2 else 0 for row in range(len(data))]

    row = first_row(build_label_prevalence_summary(data, label_columns=["label_outperform_20d"]))

    assert row["class_balance"] == "sparse positive"
    assert row["interpretation"] == "This label is sparse in this sample."


def test_label_prevalence_identifies_over_common_positive_label() -> None:
    data = panel()
    data["label_outperform_20d"] = [0] + [1] * (len(data) - 1)

    row = first_row(build_label_prevalence_summary(data, label_columns=["label_outperform_20d"]))

    assert row["class_balance"] == "highly common positive"
    assert row["interpretation"] == "This label is highly common in this sample."


def test_label_prevalence_identifies_single_class_label() -> None:
    data = panel()
    data["label_outperform_20d"] = 0

    row = first_row(build_label_prevalence_summary(data, label_columns=["label_outperform_20d"]))

    assert row["positive_rate"] == pytest.approx(0.0)
    assert row["class_balance"] == "single class"
    assert "single observed class" in row["interpretation"]


def test_label_prevalence_handles_missing_label_column() -> None:
    summary = build_label_prevalence_summary(panel(), label_columns=["missing_label"])

    assert summary.empty


def test_label_prevalence_handles_empty_input() -> None:
    summary = build_label_prevalence_summary(pd.DataFrame())

    assert summary.empty


def test_label_prevalence_handles_nan_heavy_input() -> None:
    data = panel()
    data["label_outperform_20d"] = pd.NA

    row = first_row(build_label_prevalence_summary(data, label_columns=["label_outperform_20d"]))

    assert row["sample_size"] == 0
    assert row["missing_count"] == len(data)
    assert row["class_balance"] == "insufficient"
    assert row["interpretation"] == "The sample is too small for reliable label audit."


def test_return_threshold_sensitivity_uses_forward_excess_returns() -> None:
    sensitivity = build_return_label_threshold_sensitivity(panel())

    assert sensitivity["threshold"].tolist() == [0.00, 0.01, 0.02, 0.03, 0.05]
    assert sensitivity["target_family"].eq("outperformance").all()
    assert sensitivity.loc[sensitivity["threshold"] == 0.02, "positive_count"].iloc[0] == 10
    assert sensitivity["interpretation"].str.contains("Threshold sensitivity is high").all()


def test_drawdown_threshold_sensitivity_uses_forward_drawdowns() -> None:
    sensitivity = build_drawdown_label_threshold_sensitivity(panel())

    assert sensitivity["threshold"].tolist() == [-0.05, -0.10, -0.15, -0.20]
    assert sensitivity["target_family"].eq("drawdown_risk").all()
    assert sensitivity.loc[sensitivity["threshold"] == -0.10, "positive_count"].iloc[0] == 20
    assert sensitivity["interpretation"].str.contains("Threshold sensitivity is high").all()


def distribution_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, regime, positive in (
        ("AAA", "Uptrend / low volatility", 1),
        ("BBB", "Downtrend / high risk", 0),
    ):
        for offset in range(25):
            rows.append(
                {
                    "Date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(len(rows)),
                    "Ticker": ticker,
                    "regime": regime,
                    "label_outperform_20d": positive,
                    "label_drawdown_risk_20d": 1 - positive,
                    "forward_20d_excess_return": 0.03 if positive else -0.03,
                    "forward_20d_drawdown": -0.15 if not positive else -0.03,
                }
            )
    return pd.DataFrame(rows)


def test_ticker_level_distribution_flags_material_variation() -> None:
    distribution = build_label_distribution(
        distribution_panel(),
        "Ticker",
        group_dimension="ticker",
        label_columns=["label_outperform_20d"],
    )

    assert set(distribution["group"]) == {"AAA", "BBB"}
    assert set(distribution["positive_rate"]) == {0.0, 1.0}
    assert distribution["interpretation"].str.contains("vary materially by ticker").all()


def test_regime_level_distribution_flags_material_variation() -> None:
    distribution = build_label_distribution(
        distribution_panel(),
        "regime",
        group_dimension="regime",
        label_columns=["label_outperform_20d"],
    )

    assert set(distribution["group"]) == {"Uptrend / low volatility", "Downtrend / high risk"}
    assert distribution["interpretation"].str.contains("vary materially by regime").all()


def test_return_vs_drawdown_label_overlap() -> None:
    overlap = build_return_drawdown_label_overlap(panel())

    assert {"outperform_label", "drawdown_risk_label", "sample_size", "share_of_total"}.issubset(
        overlap.columns
    )
    assert overlap["sample_size"].sum() == 30
    assert overlap["share_of_total"].sum() == pytest.approx(1.0)


def test_build_ml_label_audit_returns_expected_tables() -> None:
    audit = build_ml_label_audit(distribution_panel())

    assert not audit.prevalence_summary.empty
    assert not audit.return_threshold_sensitivity.empty
    assert not audit.drawdown_threshold_sensitivity.empty
    assert not audit.ticker_distribution.empty
    assert not audit.regime_distribution.empty
    assert not audit.label_overlap.empty


def test_label_audit_interpretations_avoid_trading_action_words() -> None:
    audit = build_ml_label_audit(distribution_panel())
    tables = [
        audit.prevalence_summary,
        audit.return_threshold_sensitivity,
        audit.drawdown_threshold_sensitivity,
        audit.ticker_distribution,
        audit.regime_distribution,
        audit.label_overlap,
    ]
    text = " ".join(
        str(value)
        for table in tables
        for value in table.get("interpretation", pd.Series(dtype=str)).dropna()
    ).lower()

    for pattern in FORBIDDEN_PATTERNS:
        assert re.search(pattern, text) is None
