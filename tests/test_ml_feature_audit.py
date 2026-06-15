from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import (
    build_feature_family_summary,
    build_feature_importance_summary,
    build_feature_redundancy_summary,
    build_ml_feature_audit,
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


class ImportanceModel:
    feature_importances_ = [0.2, 0.7, 0.1]


class CoefficientModel:
    coef_ = [[-0.3, 0.9, 0.1]]


def feature_panel(rows: int = 40) -> pd.DataFrame:
    index = range(rows)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Ticker": ["AAA" if row < rows / 2 else "BBB" for row in index],
            "return_20d": [row / 100 for row in index],
            "dist_ma_50d": [row / 200 for row in index],
            "volatility_20d": [0.2 + row / 1000 for row in index],
            "rs_spy_60d": [0.1 + row / 900 for row in index],
            "volume_z_20d": [row % 7 for row in index],
            "rsi_14d": [30 + row % 40 for row in index],
            "fourier_1": [row / 80 for row in index],
            "wavelet_1": [row / 90 for row in index],
            "label_outperform_20d": [row % 2 for row in index],
        }
    )


def test_feature_inventory_summary_normal_input() -> None:
    features = ["return_20d", "dist_ma_50d", "volatility_20d", "rs_spy_60d"]

    audit = build_ml_feature_audit(feature_panel(), features)
    row = audit.inventory_summary.iloc[0]

    assert row["feature_count"] == 4
    assert row["numeric_feature_count"] == 4
    assert row["non_numeric_feature_count"] == 0
    assert row["sample_size"] == 40
    assert row["sample_to_feature_ratio"] == pytest.approx(10.0)
    assert row["mean_missing_rate"] == pytest.approx(0.0)
    assert row["high_missing_feature_count"] == 0
    assert row["constant_or_near_constant_feature_count"] == 0


def test_feature_inventory_flags_high_missingness() -> None:
    data = feature_panel()
    data.loc[:20, "return_20d"] = pd.NA

    audit = build_ml_feature_audit(data, ["return_20d", "dist_ma_50d"])

    assert audit.inventory_summary.loc[0, "high_missing_feature_count"] == 1
    assert "high missingness" in set(audit.warnings["warning"])


def test_feature_inventory_flags_constant_and_near_constant_features() -> None:
    data = feature_panel()
    data["constant_feature"] = 1.0
    data["near_constant_feature"] = [1.0] * 39 + [2.0]

    audit = build_ml_feature_audit(data, ["constant_feature", "near_constant_feature"])

    assert audit.inventory_summary.loc[0, "constant_or_near_constant_feature_count"] == 2
    assert "constant or near-constant features" in set(audit.warnings["warning"])


def test_feature_inventory_flags_low_sample_to_feature_ratio() -> None:
    data = feature_panel(rows=5)
    features = ["return_20d", "dist_ma_50d", "volatility_20d", "rs_spy_60d"]

    audit = build_ml_feature_audit(data, features)

    warning = audit.warnings[audit.warnings["warning"] == "low sample-to-feature ratio"].iloc[0]
    assert warning["feature_count"] == 4
    assert "1.25 samples per feature" in warning["detail"]


def test_feature_family_grouping_uses_simple_feature_names() -> None:
    summary = build_feature_family_summary(
        [
            "return_20d",
            "dist_ma_200d",
            "volatility_20d",
            "rs_spy_60d",
            "volume_z_20d",
            "rsi_14d",
            "fourier_1",
            "mystery_signal",
        ]
    )

    assert set(summary["family"]) == {
        "momentum / return",
        "trend / moving average",
        "volatility",
        "relative strength / benchmark-relative",
        "volume / liquidity",
        "RSI / technical",
        "Fourier / wavelet / complex transform",
        "unknown / other",
    }


def test_feature_audit_flags_family_concentration_and_complex_sample_warning() -> None:
    data = feature_panel(rows=20)
    features = ["fourier_1", "fourier_2", "wavelet_1", "wavelet_2"]
    for column in features:
        data[column] = range(len(data))

    audit = build_ml_feature_audit(data, features)

    assert "feature family concentration" in set(audit.warnings["warning"])
    assert "complex transform features with limited sample" in set(audit.warnings["warning"])


def test_feature_redundancy_detects_high_correlation_pairs() -> None:
    data = feature_panel()
    data["return_clone"] = data["return_20d"] * 2

    summary, pairs = build_feature_redundancy_summary(data, ["return_20d", "return_clone", "volatility_20d"])

    assert summary.loc[0, "high_correlation_pair_count"] >= 1
    assert {pairs.loc[0, "feature_1"], pairs.loc[0, "feature_2"]} == {"return_20d", "return_clone"}
    assert pairs.loc[0, "abs_correlation"] == pytest.approx(1.0)


def test_feature_importance_uses_tree_importances() -> None:
    summary = build_feature_importance_summary(ImportanceModel(), ["a", "b", "c"])

    assert summary["feature"].tolist() == ["b", "a", "c"]
    assert summary["source"].eq("feature_importances_").all()


def test_feature_importance_uses_coefficients() -> None:
    summary = build_feature_importance_summary(CoefficientModel(), ["a", "b", "c"])

    assert summary["feature"].tolist() == ["b", "a", "c"]
    assert summary["source"].eq("coef_").all()


def test_feature_importance_unavailable_returns_empty_table() -> None:
    assert build_feature_importance_summary(None, ["a", "b"]).empty


def test_feature_audit_handles_empty_input() -> None:
    audit = build_ml_feature_audit(pd.DataFrame(), ["return_20d"])

    assert audit.inventory_summary.loc[0, "sample_size"] == 0
    assert audit.family_summary.loc[0, "family"] == "momentum / return"
    assert audit.warnings.empty
    assert audit.high_correlation_pairs.empty


def test_feature_audit_handles_missing_feature_columns() -> None:
    audit = build_ml_feature_audit(feature_panel(), ["return_20d", "missing_feature"])

    assert audit.inventory_summary.loc[0, "missing_feature_count"] == 1
    warning = audit.warnings[audit.warnings["warning"] == "missing feature columns"].iloc[0]
    assert warning["feature_count"] == 1
    assert warning["detail"] == "missing_feature"


def test_feature_audit_handles_nan_heavy_input() -> None:
    data = feature_panel()
    data["return_20d"] = pd.NA

    audit = build_ml_feature_audit(data, ["return_20d", "dist_ma_50d"])

    assert audit.inventory_summary.loc[0, "high_missing_feature_count"] == 1
    assert audit.inventory_summary.loc[0, "constant_or_near_constant_feature_count"] == 1


def test_feature_audit_interpretations_avoid_trading_action_words() -> None:
    audit = build_ml_feature_audit(
        feature_panel(rows=10),
        ["return_20d", "dist_ma_50d", "fourier_1", "wavelet_1"],
        model=ImportanceModel(),
    )
    tables = [
        audit.inventory_summary,
        audit.family_summary,
        audit.warnings,
        audit.redundancy_summary,
        audit.high_correlation_pairs,
        audit.redundancy_selection_summary,
        audit.redundancy_selection_report,
        audit.feature_importance,
    ]
    text = " ".join(
        str(value)
        for table in tables
        for value in table.get("interpretation", pd.Series(dtype=str)).dropna()
    ).lower()

    for pattern in FORBIDDEN_PATTERNS:
        assert re.search(pattern, text) is None
