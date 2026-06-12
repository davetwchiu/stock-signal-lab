from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import (
    build_ml_diagnostics,
    build_ml_score_direction_diagnostics,
    build_probability_label_alignment,
    build_regime_score_direction_summary,
    build_score_bucket_monotonicity,
    build_score_inversion_diagnostics,
)


def validation_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    outperformance = pd.DataFrame(
        {
            "fold": [1, 1, 1, 2, 2, 2],
            "Date": dates,
            "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "actual": [0, 0, 0, 1, 1, 1],
            "probability": [0.90, 0.85, 0.65, 0.60, 0.30, 0.20],
            "prediction": [0, 0, 1, 1, 1, 1],
            "forward_return": [-0.05, -0.02, 0.01, 0.03, 0.08, 0.10],
            "forward_excess_return": [-0.08, -0.04, -0.01, 0.02, 0.06, 0.09],
            "forward_drawdown": [-0.20, -0.15, -0.08, -0.07, -0.04, -0.03],
        }
    )
    drawdown_risk = pd.DataFrame(
        {
            "fold": [1, 1, 1, 2, 2, 2],
            "Date": dates,
            "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "actual": [1, 1, 0, 0, 0, 0],
            "probability": [0.80, 0.75, 0.45, 0.35, 0.20, 0.10],
            "prediction": [1, 1, 0, 0, 0, 0],
        }
    )
    return outperformance, drawdown_risk


def test_ml_diagnostics_bucket_existing_out_of_sample_scores() -> None:
    outperformance, drawdown_risk = validation_predictions()

    diagnostics = build_ml_diagnostics(outperformance, drawdown_risk)
    buckets = diagnostics.score_buckets.set_index("score_bucket")

    assert buckets.loc["Low", "count"] == 2
    assert buckets.loc["Medium", "count"] == 2
    assert buckets.loc["High", "count"] == 2
    assert buckets.loc["Low", "outperformance_hit_rate"] == 0.0
    assert buckets.loc["High", "outperformance_hit_rate"] == 1.0
    assert (
        buckets.loc["High", "average_forward_excess_return"]
        > buckets.loc["Low", "average_forward_excess_return"]
    )
    assert buckets.loc["Low", "average_drawdown_risk_probability"] > buckets.loc[
        "High", "average_drawdown_risk_probability"
    ]


def test_ml_diagnostics_include_drawdown_risk_calibration_and_summary() -> None:
    outperformance, drawdown_risk = validation_predictions()
    out_metrics = pd.DataFrame([{"accuracy": 0.5, "roc_auc": 0.7}])
    risk_metrics = pd.DataFrame([{"accuracy": 0.8, "roc_auc": 0.75}])

    diagnostics = build_ml_diagnostics(outperformance, drawdown_risk, out_metrics, risk_metrics, risk_bins=2)

    assert list(diagnostics.summary["target"]) == ["outperformance", "drawdown_risk"]
    assert diagnostics.summary.loc[0, "predictions"] == 6
    assert diagnostics.summary.loc[0, "folds"] == 2
    assert diagnostics.summary.loc[1, "accuracy"] == 0.8
    assert "observed_drawdown_risk_rate" in diagnostics.drawdown_risk_calibration.columns
    assert diagnostics.drawdown_risk_calibration["count"].sum() == 6


def test_ml_diagnostics_handle_empty_predictions() -> None:
    diagnostics = build_ml_diagnostics(pd.DataFrame(), pd.DataFrame())

    assert diagnostics.score_buckets.empty
    assert diagnostics.drawdown_risk_calibration.empty
    assert list(diagnostics.summary["target"]) == ["outperformance", "drawdown_risk"]
    assert diagnostics.summary["predictions"].tolist() == [0, 0]


def score_direction_panel(
    *,
    low_return: float = 0.00,
    medium_return: float = 0.03,
    high_return: float = 0.08,
    rows_per_bucket: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    date_index = pd.date_range("2024-01-01", periods=rows_per_bucket * 3, freq="B")
    row_index = 0
    for bucket, score, forward_return, probability, return_label, risk_label, drawdown in (
        ("low", 20.0, low_return, 0.20, 0, 1, -0.18),
        ("medium", 55.0, medium_return, 0.55, 1, 0, -0.10),
        ("high", 85.0, high_return, 0.85, 1, 0, -0.04),
    ):
        for bucket_index in range(rows_per_bucket):
            rows.append(
                {
                    "Date": date_index[row_index],
                    "Ticker": f"{bucket[:1].upper()}{bucket_index:02d}",
                    "ML Score": score,
                    "probability_out": probability,
                    "probability_risk": 1.0 - probability,
                    "actual_out": return_label,
                    "actual_risk": risk_label,
                    "forward_return": forward_return,
                    "forward_drawdown": drawdown,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def test_score_direction_diagnostics_identify_aligned_direction() -> None:
    summary = build_ml_score_direction_diagnostics(score_direction_panel())

    row = summary.iloc[0]
    assert row["sample_size"] == 30
    assert row["score_column"] == "ML Score"
    assert row["target_column"] == "forward_return"
    assert row["label_column"] == "actual_out"
    assert row["drawdown_label_column"] == "actual_risk"
    assert row["top_minus_bottom_spread"] == pytest.approx(0.08)
    assert row["score_to_forward_return_spearman"] > 0
    assert row["score_to_return_label_spearman"] > 0
    assert row["score_to_drawdown_label_spearman"] < 0
    assert row["direction"] == "aligned"


def test_score_direction_diagnostics_identify_inverted_direction() -> None:
    summary = build_ml_score_direction_diagnostics(
        score_direction_panel(low_return=0.08, medium_return=0.03, high_return=0.00)
    )

    assert summary.iloc[0]["top_minus_bottom_spread"] == pytest.approx(-0.08)
    assert summary.iloc[0]["direction"] == "inverted"


def test_score_direction_diagnostics_identify_flat_direction() -> None:
    summary = build_ml_score_direction_diagnostics(
        score_direction_panel(low_return=0.02, medium_return=0.02, high_return=0.02)
    )

    assert summary.iloc[0]["top_minus_bottom_spread"] == pytest.approx(0.0)
    assert summary.iloc[0]["direction"] == "flat"


def test_score_direction_diagnostics_handle_insufficient_sample() -> None:
    summary = build_ml_score_direction_diagnostics(score_direction_panel(rows_per_bucket=2))

    assert summary.iloc[0]["direction"] == "insufficient"
    assert "too small" in summary.iloc[0]["interpretation"]


def test_probability_label_alignment_shows_score_relationships() -> None:
    alignment = build_probability_label_alignment(score_direction_panel()).set_index("diagnostic")

    assert alignment.loc[
        "positive return probability",
        "higher_score_corresponds_to",
    ] == "higher positive-return probability"
    assert alignment.loc[
        "realised forward return",
        "higher_score_corresponds_to",
    ] == "higher realised forward return"
    assert alignment.loc[
        "drawdown-risk label rate",
        "higher_score_corresponds_to",
    ] == "lower drawdown-risk label rate"
    assert alignment.loc[
        "realised drawdown event rate",
        "higher_score_corresponds_to",
    ] == "lower realised drawdown event rate"
    assert alignment.loc[
        "realised forward drawdown",
        "higher_score_corresponds_to",
    ] == "less severe realised drawdowns"


def test_score_bucket_monotonicity_reports_bucket_outcomes() -> None:
    monotonicity = build_score_bucket_monotonicity(score_direction_panel())

    assert monotonicity["bucket"].tolist() == ["Low", "Medium", "High"]
    assert monotonicity["monotonicity_result"].unique().tolist() == ["aligned"]
    assert monotonicity.loc[0, "return_label_rate"] == pytest.approx(0.0)
    assert monotonicity.loc[2, "drawdown_label_rate"] == pytest.approx(0.0)


def test_score_bucket_monotonicity_identifies_inverted_buckets() -> None:
    monotonicity = build_score_bucket_monotonicity(
        score_direction_panel(low_return=0.08, medium_return=0.03, high_return=0.00)
    )

    assert monotonicity["monotonicity_result"].unique().tolist() == ["inverted"]


def test_score_inversion_diagnostics_compare_current_and_inverted_direction() -> None:
    inversion = build_score_inversion_diagnostics(
        score_direction_panel(low_return=0.08, medium_return=0.03, high_return=0.00)
    ).set_index("score_direction")

    assert inversion.loc["current ML score", "top_minus_bottom_spread"] == pytest.approx(-0.08)
    assert inversion.loc["inverted score", "top_minus_bottom_spread"] == pytest.approx(0.08)
    assert inversion.loc[
        "current ML score",
        "better_forward_return_separation",
    ] == "inverted score"


def test_score_direction_helpers_handle_missing_target_or_label_columns() -> None:
    panel = score_direction_panel()

    missing_target = build_ml_score_direction_diagnostics(panel.drop(columns=["forward_return"]))
    missing_label_alignment = build_probability_label_alignment(panel.drop(columns=["actual_out", "actual_risk"]))
    missing_monotonicity = build_score_bucket_monotonicity(panel.drop(columns=["forward_return"]))

    assert missing_target.iloc[0]["direction"] == "insufficient"
    assert pd.isna(missing_target.iloc[0]["target_column"])
    assert "positive return label rate" not in set(missing_label_alignment["diagnostic"])
    assert "drawdown-risk label rate" not in set(missing_label_alignment["diagnostic"])
    assert "realised drawdown event rate" not in set(missing_label_alignment["diagnostic"])
    assert missing_monotonicity.empty


def test_regime_score_direction_summary_uses_existing_regime_labels() -> None:
    panel = score_direction_panel()
    baseline = pd.DataFrame(
        {
            "Date": panel["Date"],
            "Ticker": panel["Ticker"],
            "regime": ["Uptrend"] * 15 + ["Downtrend"] * 15,
        }
    )

    summary = build_regime_score_direction_summary(panel, baseline_panel=baseline, min_samples=10)

    assert set(summary["regime"]) == {"Uptrend", "Downtrend"}
    assert set(summary["direction"]).issubset({"aligned", "inverted", "flat", "insufficient"})


def test_build_ml_diagnostics_includes_score_direction_tables() -> None:
    outperformance, drawdown_risk = validation_predictions()

    diagnostics = build_ml_diagnostics(outperformance, drawdown_risk, risk_bins=2)

    assert not diagnostics.score_direction_summary.empty
    assert not diagnostics.probability_label_alignment.empty
    assert not diagnostics.score_bucket_monotonicity.empty
    assert not diagnostics.score_inversion.empty


def test_score_direction_interpretations_avoid_trading_action_words() -> None:
    panel = score_direction_panel(low_return=0.08, medium_return=0.03, high_return=0.00)
    frames = [
        build_ml_score_direction_diagnostics(panel),
        build_probability_label_alignment(panel),
        build_score_bucket_monotonicity(panel),
        build_score_inversion_diagnostics(panel),
        build_regime_score_direction_summary(panel.assign(regime="Test regime"), min_samples=10),
    ]
    text = " ".join(
        " ".join(frame["interpretation"].dropna().astype(str))
        for frame in frames
        if "interpretation" in frame
    ).lower()

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
