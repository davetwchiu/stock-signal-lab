from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import (
    _risk_haircut_score,
    build_opportunity_risk_joint_validation,
    build_ml_diagnostics,
    build_ml_probability_direction_check,
    build_ml_score_formula_candidate_comparison,
    build_ml_score_direction_diagnostics,
    build_ml_target_comparison,
    interpret_ml_probability_direction_check,
    interpret_ml_score_formula_candidate_comparison,
    interpret_opportunity_risk_joint_validation,
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
            "probability": [0.10, 0.20, 0.45, 0.50, 0.75, 0.85],
            "prediction": [0, 0, 0, 1, 1, 1],
            "forward_return": [-0.05, -0.02, 0.01, 0.03, 0.08, 0.10],
            "forward_excess_return": [-0.08, -0.04, -0.01, 0.02, 0.06, 0.09],
            "forward_drawdown": [-0.20, -0.15, -0.08, -0.07, -0.04, -0.03],
            "forward_risk_adjusted_excess_return": [-0.18, -0.115, -0.05, -0.015, 0.04, 0.075],
        }
    )
    drawdown_risk = pd.DataFrame(
        {
            "fold": [1, 1, 1, 2, 2, 2],
            "Date": dates,
            "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "actual": [1, 1, 0, 0, 0, 0],
            "probability": [0.80, 0.75, 0.50, 0.45, 0.20, 0.10],
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


def test_ml_diagnostics_merge_overlapping_folds_without_row_explosion() -> None:
    date = pd.Timestamp("2024-01-02")
    outperformance = pd.DataFrame(
        {
            "fold": [1, 2],
            "Date": [date, date],
            "Ticker": ["AAA", "AAA"],
            "actual": [0, 1],
            "probability": [0.20, 0.80],
            "forward_return": [0.01, 0.02],
            "forward_excess_return": [-0.01, 0.04],
            "forward_drawdown": [-0.08, -0.04],
        }
    )
    drawdown_risk = pd.DataFrame(
        {
            "fold": [1, 2],
            "Date": [date, date],
            "Ticker": ["AAA", "AAA"],
            "actual": [1, 0],
            "probability": [0.70, 0.20],
        }
    )

    diagnostics = build_ml_diagnostics(outperformance, drawdown_risk, risk_bins=2)

    assert diagnostics.score_buckets["count"].sum() == 2
    assert diagnostics.drawdown_risk_calibration["count"].sum() == 2


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
                    "forward_excess_return": forward_return,
                    "forward_drawdown": drawdown,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def probability_direction_panel(
    *,
    raw_direction: str = "supported",
    rows_per_bucket: int = 10,
    include_risk: bool = True,
    constant_probability: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-03-01", periods=rows_per_bucket * 3, freq="B")
    if raw_direction == "supported":
        returns = (0.00, 0.03, 0.08)
        labels = (0, 1, 1)
    elif raw_direction == "inverted":
        returns = (0.08, 0.03, 0.00)
        labels = (1, 1, 0)
    else:
        returns = (0.02, 0.02, 0.02)
        labels = (1, 1, 1)

    row_index = 0
    for probability, risk, forward_excess_return, actual in zip(
        (0.20, 0.50, 0.80),
        (0.80, 0.50, 0.10),
        returns,
        labels,
        strict=True,
    ):
        for bucket_index in range(rows_per_bucket):
            row = {
                "Date": dates[row_index],
                "Ticker": f"P{row_index:02d}",
                "probability_out": 0.50 if constant_probability else probability + bucket_index * 0.0001,
                "actual_out": actual,
                "forward_excess_return": forward_excess_return,
                "forward_return": forward_excess_return + 0.01,
            }
            if include_risk:
                row["probability_risk"] = risk + bucket_index * 0.0001
            rows.append(row)
            row_index += 1
    return pd.DataFrame(rows)


def test_probability_direction_check_identifies_raw_direction() -> None:
    direction = build_ml_probability_direction_check(probability_direction_panel())

    indexed = direction.set_index("signal")
    assert indexed.loc["raw probability", "monotonicity"] == "aligned"
    assert indexed.loc["raw probability", "high_minus_low_spread"] == pytest.approx(0.08)
    assert indexed.loc["current ML Score", "monotonicity"] == "aligned"
    assert indexed.loc["raw probability", "actual_label_rate_high_bucket"] > indexed.loc[
        "raw probability",
        "actual_label_rate_low_bucket",
    ]
    assert (
        interpret_ml_probability_direction_check(direction)
        == "raw outperformance probability direction is supported"
    )


def test_probability_direction_check_identifies_inverted_direction() -> None:
    direction = build_ml_probability_direction_check(probability_direction_panel(raw_direction="inverted"))

    indexed = direction.set_index("signal")
    assert indexed.loc["inverted probability", "monotonicity"] == "aligned"
    assert indexed.loc["inverted probability", "high_minus_low_spread"] == pytest.approx(0.08)
    assert (
        interpret_ml_probability_direction_check(direction)
        == "inverted outperformance probability direction is supported"
    )


def test_probability_direction_check_does_not_make_current_score_direction_from_risk_only() -> None:
    panel = probability_direction_panel(raw_direction="supported", constant_probability=True)

    direction = build_ml_probability_direction_check(panel)

    indexed = direction.set_index("signal")
    assert indexed.loc["raw probability", "monotonicity"] == "insufficient"
    assert indexed.loc["current ML Score", "monotonicity"] == "insufficient"
    assert interpret_ml_probability_direction_check(direction) == "insufficient data"


def test_probability_direction_check_identifies_corrected_current_ml_score_direction() -> None:
    panel = probability_direction_panel(raw_direction="supported", include_risk=False)
    panel["probability_risk"] = 0.30

    direction = build_ml_probability_direction_check(panel)

    indexed = direction.set_index("signal")
    assert indexed.loc["current ML Score", "monotonicity"] == "aligned"
    assert indexed.loc["current ML Score", "high_minus_low_spread"] == pytest.approx(0.08)


def test_probability_direction_check_returns_mixed_for_flat_data() -> None:
    direction = build_ml_probability_direction_check(probability_direction_panel(raw_direction="flat"))

    assert set(direction["monotonicity"]) == {"flat"}
    assert interpret_ml_probability_direction_check(direction) == "direction evidence is mixed"


def test_probability_direction_check_handles_insufficient_data() -> None:
    direction = build_ml_probability_direction_check(probability_direction_panel(rows_per_bucket=1))

    assert set(direction["monotonicity"]) == {"insufficient"}
    assert interpret_ml_probability_direction_check(direction) == "insufficient data"


def test_probability_direction_check_allows_missing_risk_probability() -> None:
    direction = build_ml_probability_direction_check(probability_direction_panel(include_risk=False))

    assert direction["signal"].tolist() == ["raw probability", "inverted probability"]
    assert (
        interpret_ml_probability_direction_check(direction)
        == "raw outperformance probability direction is supported"
    )


def test_probability_direction_check_does_not_mutate_input() -> None:
    panel = probability_direction_panel()
    original = panel.copy(deep=True)

    build_ml_probability_direction_check(panel)

    pd.testing.assert_frame_equal(panel, original)


def formula_candidate_panel(
    *,
    rows_per_bucket: int = 10,
    flat: bool = False,
    include_drawdown_label: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-04-01", periods=rows_per_bucket * 3, freq="B")
    scenarios = [
        (0.30, 0.00, 0.00, 0, 0),
        (0.40, 0.50, 0.03, 1, 0),
        (0.50, 1.00, 0.08, 1, 1),
    ]
    row_index = 0
    for probability, risk, forward_excess_return, actual, drawdown_event in scenarios:
        for bucket_index in range(rows_per_bucket):
            row = {
                "Date": dates[row_index],
                "Ticker": f"F{row_index:02d}",
                "probability_out": 0.40 if flat else probability + bucket_index * 0.0001,
                "probability_risk": 0.50 if flat else risk,
                "actual_out": actual,
                "forward_excess_return": forward_excess_return,
            }
            if include_drawdown_label:
                row["actual_risk"] = drawdown_event
            rows.append(row)
            row_index += 1
    return pd.DataFrame(rows)


def test_formula_candidate_comparison_includes_expected_candidate_rows() -> None:
    comparison = build_ml_score_formula_candidate_comparison(formula_candidate_panel(include_drawdown_label=False))

    assert comparison["candidate_name"].tolist() == [
        "raw_probability",
        "sqrt_opportunity_only",
        "current_production_score",
        "light_risk_penalty",
        "risk_haircut_score",
    ]


def test_formula_candidate_comparison_uses_quantile_low_mid_high_groups() -> None:
    comparison = build_ml_score_formula_candidate_comparison(formula_candidate_panel(include_drawdown_label=False))
    raw = comparison.set_index("candidate_name").loc["raw_probability"]

    assert raw["low_bucket_forward_excess_return"] == pytest.approx(0.00)
    assert raw["mid_bucket_forward_excess_return"] == pytest.approx(0.03)
    assert raw["high_bucket_forward_excess_return"] == pytest.approx(0.08)
    assert raw["monotonicity"] == "aligned"


def test_formula_candidate_comparison_identifies_opportunity_only_as_strong() -> None:
    comparison = build_ml_score_formula_candidate_comparison(formula_candidate_panel(include_drawdown_label=False))
    indexed = comparison.set_index("candidate_name")

    assert indexed.loc["raw_probability", "interpretation"] == "candidate looks strong"
    assert indexed.loc["sqrt_opportunity_only", "interpretation"] == "candidate looks strong"
    assert interpret_ml_score_formula_candidate_comparison(comparison) == "formula evidence is mixed"


def test_formula_candidate_comparison_current_score_matches_raw_probability() -> None:
    comparison = build_ml_score_formula_candidate_comparison(formula_candidate_panel(include_drawdown_label=False))
    indexed = comparison.set_index("candidate_name")
    raw = indexed.loc["raw_probability"]
    current = indexed.loc["current_production_score"]

    assert current["low_bucket_forward_excess_return"] == pytest.approx(raw["low_bucket_forward_excess_return"])
    assert current["mid_bucket_forward_excess_return"] == pytest.approx(raw["mid_bucket_forward_excess_return"])
    assert current["high_bucket_forward_excess_return"] == pytest.approx(raw["high_bucket_forward_excess_return"])
    assert current["high_minus_low_spread"] == pytest.approx(raw["high_minus_low_spread"])
    assert current["monotonicity"] == "aligned"
    assert current["interpretation"] == "candidate looks strong"


def test_formula_candidate_comparison_computes_risk_haircut_score() -> None:
    opportunity_score = pd.Series([50.0, 50.0, 50.0])
    risk_probability = pd.Series([0.39, 0.40, 0.60])

    score = _risk_haircut_score(opportunity_score, risk_probability)

    assert score.tolist() == pytest.approx([50.0, 42.5, 35.0])


def test_formula_candidate_comparison_handles_insufficient_flat_and_missing_data() -> None:
    flat = build_ml_score_formula_candidate_comparison(formula_candidate_panel(flat=True))
    missing = build_ml_score_formula_candidate_comparison(
        formula_candidate_panel().drop(columns=["forward_excess_return"])
    )
    tiny = build_ml_score_formula_candidate_comparison(formula_candidate_panel(rows_per_bucket=1))

    assert set(flat["interpretation"]) == {"insufficient data"}
    assert missing.empty
    assert set(tiny["interpretation"]) == {"insufficient data"}
    assert interpret_ml_score_formula_candidate_comparison(flat) == "insufficient data"


def test_formula_candidate_comparison_does_not_mutate_input() -> None:
    panel = formula_candidate_panel()
    original = panel.copy(deep=True)

    build_ml_score_formula_candidate_comparison(panel)

    pd.testing.assert_frame_equal(panel, original)


def test_score_direction_diagnostics_identify_aligned_direction() -> None:
    summary = build_ml_score_direction_diagnostics(score_direction_panel())

    row = summary.iloc[0]
    assert row["sample_size"] == 30
    assert row["score_column"] == "ML Score"
    assert row["target_column"] == "forward_excess_return"
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
        "outperformance probability",
        "higher_score_corresponds_to",
    ] == "higher outperformance probability"
    assert alignment.loc[
        "realised forward excess return",
        "higher_score_corresponds_to",
    ] == "higher realised forward excess return"
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

    missing_target = build_ml_score_direction_diagnostics(panel.drop(columns=["forward_excess_return"]))
    missing_label_alignment = build_probability_label_alignment(panel.drop(columns=["actual_out", "actual_risk"]))
    missing_monotonicity = build_score_bucket_monotonicity(panel.drop(columns=["forward_excess_return"]))

    assert missing_target.iloc[0]["direction"] == "insufficient"
    assert pd.isna(missing_target.iloc[0]["target_column"])
    assert "outperformance label rate" not in set(missing_label_alignment["diagnostic"])
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
    assert not diagnostics.probability_direction_check.empty
    assert not diagnostics.formula_candidate_comparison.empty


def opportunity_risk_panel(rows_per_cell: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-02-01", periods=rows_per_cell * 4, freq="B")
    scenarios = [
        ("High opportunity", "Low risk", 0.90, 0.10, 0.08, -0.04, 0),
        ("High opportunity", "High risk", 0.80, 0.90, 0.05, -0.18, 1),
        ("Low opportunity", "Low risk", 0.20, 0.20, 0.01, -0.05, 0),
        ("Low opportunity", "High risk", 0.10, 0.80, -0.03, -0.22, 1),
    ]
    row_index = 0
    for opportunity_bucket, risk_bucket, opportunity, risk, forward_return, drawdown, risk_label in scenarios:
        for cell_index in range(rows_per_cell):
            rows.append(
                {
                    "Date": dates[row_index],
                    "Ticker": f"J{row_index:02d}",
                    "probability_out": opportunity - cell_index * 0.001,
                    "probability_risk": risk + cell_index * 0.001,
                    "forward_excess_return": forward_return,
                    "forward_drawdown": drawdown,
                    "actual_risk": risk_label,
                    "expected_opportunity_bucket": opportunity_bucket,
                    "expected_risk_bucket": risk_bucket,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def test_opportunity_risk_joint_validation_includes_expected_four_cells() -> None:
    matrix = build_opportunity_risk_joint_validation(opportunity_risk_panel())

    cells = set(zip(matrix["opportunity_bucket"], matrix["risk_bucket"], strict=True))

    assert cells == {
        ("High opportunity", "Low risk"),
        ("High opportunity", "High risk"),
        ("Low opportunity", "Low risk"),
        ("Low opportunity", "High risk"),
    }
    assert matrix["sample_size"].tolist() == [5, 5, 5, 5]


def test_opportunity_risk_joint_validation_identifies_best_supported_cell() -> None:
    matrix = build_opportunity_risk_joint_validation(opportunity_risk_panel())
    indexed = matrix.set_index(["opportunity_bucket", "risk_bucket"])

    assert indexed.loc[("High opportunity", "Low risk"), "avg_forward_excess_return"] == pytest.approx(0.08)
    assert indexed.loc[("High opportunity", "Low risk"), "interpretation"].startswith("Best setup")
    assert (
        interpret_opportunity_risk_joint_validation(matrix)
        == "joint validation supports separate opportunity and risk signals"
    )


def test_opportunity_risk_joint_validation_flags_high_opportunity_high_risk_as_riskier() -> None:
    matrix = build_opportunity_risk_joint_validation(opportunity_risk_panel())
    indexed = matrix.set_index(["opportunity_bucket", "risk_bucket"])

    assert indexed.loc[("High opportunity", "High risk"), "avg_forward_drawdown"] < indexed.loc[
        ("High opportunity", "Low risk"),
        "avg_forward_drawdown",
    ]
    assert indexed.loc[("High opportunity", "High risk"), "drawdown_event_rate"] > indexed.loc[
        ("High opportunity", "Low risk"),
        "drawdown_event_rate",
    ]


def test_opportunity_risk_joint_validation_identifies_low_opportunity_high_risk_as_worst() -> None:
    matrix = build_opportunity_risk_joint_validation(opportunity_risk_panel())
    indexed = matrix.set_index(["opportunity_bucket", "risk_bucket"])

    assert indexed.loc[("Low opportunity", "High risk"), "avg_forward_excess_return"] == pytest.approx(-0.03)
    assert indexed.loc[("Low opportunity", "High risk"), "interpretation"].startswith("Worst setup")


def test_opportunity_risk_joint_validation_handles_insufficient_data() -> None:
    matrix = build_opportunity_risk_joint_validation(opportunity_risk_panel(rows_per_cell=1))

    assert matrix.empty
    assert interpret_opportunity_risk_joint_validation(matrix) == "insufficient data to compare"


def test_opportunity_risk_joint_validation_handles_missing_columns() -> None:
    panel = opportunity_risk_panel().drop(columns=["probability_risk"])

    matrix = build_opportunity_risk_joint_validation(panel)

    assert matrix.empty


def test_opportunity_risk_joint_validation_does_not_mutate_input() -> None:
    panel = opportunity_risk_panel()
    original = panel.copy(deep=True)

    build_opportunity_risk_joint_validation(panel)

    pd.testing.assert_frame_equal(panel, original)


def test_build_ml_diagnostics_includes_opportunity_risk_joint_validation() -> None:
    panel = opportunity_risk_panel()
    outperformance = panel.rename(
        columns={
            "probability_out": "probability",
            "expected_opportunity_bucket": "unused_opportunity_bucket",
        }
    )
    outperformance["actual"] = [1] * len(outperformance)
    drawdown_risk = panel[["Date", "Ticker", "probability_risk", "actual_risk"]].rename(
        columns={"probability_risk": "probability", "actual_risk": "actual"}
    )

    diagnostics = build_ml_diagnostics(outperformance, drawdown_risk)

    assert not diagnostics.opportunity_risk_joint_validation.empty
    assert (
        interpret_opportunity_risk_joint_validation(diagnostics.opportunity_risk_joint_validation)
        == "joint validation supports separate opportunity and risk signals"
    )


def target_comparison_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    v1_rows: list[dict[str, object]] = []
    v2_rows: list[dict[str, object]] = []
    v3_rows: list[dict[str, object]] = []
    for index, probability in enumerate([0.10] * 10 + [0.50] * 10 + [0.90] * 10):
        v1_excess = 0.00 if index < 10 else 0.02 if index < 20 else 0.03
        v2_target = -0.04 if index < 10 else 0.01 if index < 20 else 0.08
        v3_target = 0.00 if index < 10 else 0.02 if index < 20 else 0.032
        base = {
            "fold": 1,
            "Date": dates[index],
            "Ticker": f"T{index:02d}",
            "actual": 1 if probability >= 0.50 else 0,
            "probability": probability,
            "prediction": 1 if probability >= 0.50 else 0,
        }
        v1_rows.append(
            {
                **base,
                "forward_excess_return": v1_excess,
                "forward_risk_adjusted_excess_return": v2_target,
            }
        )
        v2_rows.append(
            {
                **base,
                "forward_excess_return": v1_excess,
                "forward_risk_adjusted_excess_return": v2_target,
            }
        )
        v3_rows.append(
            {
                **base,
                "forward_excess_return": v1_excess,
                "forward_tail_risk_adjusted_excess_return": v3_target,
            }
        )
    return pd.DataFrame(v1_rows), pd.DataFrame(v2_rows), pd.DataFrame(v3_rows)


def test_ml_target_comparison_includes_v1_v2_and_v3() -> None:
    v1, v2, v3 = target_comparison_predictions()

    comparison = build_ml_target_comparison(v1, v2, v3).set_index("target_version")

    assert comparison.index.tolist() == [
        "v1 outperformance",
        "v2 risk-adjusted relative",
        "v3 tail-risk relative",
    ]
    assert comparison.loc["v1 outperformance", "relative_result"] == "v1 reference"
    assert comparison.loc["v1 outperformance", "high_minus_low_spread"] == pytest.approx(0.03)
    assert comparison.loc["v2 risk-adjusted relative", "high_minus_low_spread"] == pytest.approx(0.12)
    assert comparison.loc["v2 risk-adjusted relative", "monotonicity"] == "aligned"
    assert comparison.loc["v2 risk-adjusted relative", "relative_result"] == "v2 looks better"
    assert comparison.loc["v2 risk-adjusted relative", "baseline_spread"] == pytest.approx(0.03)
    assert comparison.loc["v3 tail-risk relative", "relative_result"] == "v3 looks similar"


def test_ml_target_comparison_handles_insufficient_data() -> None:
    tiny = pd.DataFrame(
        {
            "probability": [0.2, 0.8],
            "forward_excess_return": [0.01, 0.02],
            "forward_risk_adjusted_excess_return": [0.00, 0.01],
            "forward_tail_risk_adjusted_excess_return": [0.01, 0.02],
        }
    )

    comparison = build_ml_target_comparison(tiny, tiny, tiny)

    assert comparison["relative_result"].tolist() == ["v1 reference", "insufficient data", "insufficient data"]
    assert comparison.loc[1, "monotonicity"] == "insufficient"


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
