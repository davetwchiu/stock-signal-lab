from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.datasets import build_supervised_panel
from src.ml.scoring import ml_score
from src.ml.target_diagnostics import (
    TargetCandidate,
    add_target_candidate_labels,
    build_target_arena_comparison,
    build_target_balance_diagnostics,
    build_target_feature_group_comparison,
    build_target_quality_summary,
    build_target_regime_comparison,
    build_target_stability_summary,
    build_target_walk_forward_comparison,
    target_candidate_registry,
)


def price_frame(values: list[float], index: pd.DatetimeIndex) -> pd.DataFrame:
    price = pd.Series(values, index=index, dtype=float)
    return pd.DataFrame(
        {
            "Adj Close": price,
            "daily_return": price.pct_change().fillna(0.0),
            "volatility_20d": 0.02,
            "feature_signal": np.linspace(0.0, 1.0, len(index)),
        },
        index=index,
    )


def synthetic_supervised_panel() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-01", periods=90, freq="B")
    benchmark = pd.Series(np.linspace(100.0, 115.0, len(index)), index=index)
    frames = {
        "AAA": price_frame(np.linspace(100.0, 190.0, len(index)).tolist(), index),
        "BBB": price_frame(np.linspace(100.0, 145.0, len(index)).tolist(), index),
        "CCC": price_frame(np.linspace(100.0, 105.0, len(index)).tolist(), index),
    }
    panel = build_supervised_panel(frames, benchmark, horizon=20)
    return panel, benchmark


def test_target_candidate_registry_includes_current_production_baseline() -> None:
    candidates = target_candidate_registry()

    baseline = candidates[0]
    assert baseline.target_id == "outperform_20d"
    assert baseline.label_column == "label_outperform_20d"
    assert baseline.target_type == "production_baseline"


def test_alternative_target_labels_are_created_with_expected_columns() -> None:
    panel, benchmark = synthetic_supervised_panel()

    result = add_target_candidate_labels(panel, benchmark)

    expected = {
        "label_outperform_60d",
        "label_risk_adjusted_excess_20d",
        "label_top_tercile_excess_20d",
        "label_tail_adjusted_outperform_20d",
        "label_drawdown_adjusted_opportunity_20d",
        "label_pullback_recovery_20d",
    }
    assert expected.issubset(result.columns)


def test_outperform_60d_uses_sixty_day_forward_excess_return() -> None:
    panel, benchmark = synthetic_supervised_panel()

    result = add_target_candidate_labels(panel, benchmark)
    first = result[result["Ticker"] == "AAA"].sort_values("Date").iloc[0]
    aaa = panel[panel["Ticker"] == "AAA"].sort_values("Date")
    ticker_return = aaa["Adj Close"].iloc[60] / aaa["Adj Close"].iloc[0] - 1.0
    benchmark_return = benchmark.iloc[60] / benchmark.iloc[0] - 1.0

    assert first["forward_60d_excess_return"] == pytest.approx(ticker_return - benchmark_return)
    assert first["label_outperform_60d"] == float(ticker_return - benchmark_return > 0.02)


def test_top_tercile_target_is_cross_sectional_and_skips_small_groups() -> None:
    dates = pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 2)
    panel = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB"],
            "Date": dates,
            "Adj Close": 100.0,
            "forward_20d_excess_return": [0.10, 0.02, -0.01, 0.20, -0.02],
            "forward_20d_drawdown": [-0.02, -0.02, -0.02, -0.02, -0.02],
            "label_outperform_20d": [1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    benchmark = pd.Series([100.0, 100.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    result = add_target_candidate_labels(panel, benchmark, min_cross_sectional_count=3)

    first_date = result[result["Date"] == pd.Timestamp("2024-01-02")].set_index("Ticker")
    assert first_date.loc["AAA", "label_top_tercile_excess_20d"] == 1.0
    assert first_date.loc["BBB", "label_top_tercile_excess_20d"] == 0.0
    assert result[result["Date"] == pd.Timestamp("2024-01-03")]["label_top_tercile_excess_20d"].isna().all()


def test_tail_adjusted_target_penalizes_large_forward_drawdown() -> None:
    panel = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "Adj Close": [100.0, 100.0],
            "forward_20d_excess_return": [0.04, 0.04],
            "forward_20d_drawdown": [-0.08, -0.01],
            "label_outperform_20d": [1.0, 1.0],
        }
    )
    benchmark = pd.Series([100.0], index=pd.to_datetime(["2024-01-02"]))

    result = add_target_candidate_labels(panel, benchmark, outperformance_threshold=0.02)

    indexed = result.set_index("Ticker")
    assert indexed.loc["AAA", "label_tail_adjusted_outperform_20d"] == 0.0
    assert indexed.loc["BBB", "label_tail_adjusted_outperform_20d"] == 1.0
    assert indexed.loc["AAA", "label_drawdown_adjusted_opportunity_20d"] == 0.0
    assert indexed.loc["BBB", "label_drawdown_adjusted_opportunity_20d"] == 1.0


def test_target_balance_diagnostics_include_counts_rates_and_balance_status() -> None:
    panel, benchmark = synthetic_supervised_panel()
    target_panel = add_target_candidate_labels(panel, benchmark)

    diagnostics = build_target_balance_diagnostics(target_panel, min_sample_count=10)

    row = diagnostics.set_index("target_id").loc["outperform_20d"]
    assert row["sample_count"] > 0
    assert 0.0 <= row["positive_rate"] <= 1.0
    assert row["class_balance_status"] in {"Healthy", "Skewed", "Unusable"}


def walk_forward_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    rows: list[dict[str, object]] = []
    for ticker_offset, ticker in enumerate(["AAA", "BBB"]):
        for date_index, date_value in enumerate(dates):
            label = int((date_index + ticker_offset) % 4 >= 2)
            rows.append(
                {
                    "Ticker": ticker,
                    "Date": date_value,
                    "feature_signal": float(label) + 0.1 * ticker_offset,
                    "feature_fourier": float(label) * 0.8 + 0.05 * date_index,
                    "feature_wavelet": float(label) * 0.6 + 0.02 * ticker_offset,
                    "regime": "Uptrend" if label else "Downtrend",
                    "label_custom_20d": label,
                    "forward_20d_excess_return": 0.02 if label else -0.01,
                    "forward_20d_drawdown": -0.02 if label else -0.06,
                }
            )
    return pd.DataFrame(rows)


def test_target_walk_forward_comparison_returns_one_row_per_usable_target() -> None:
    panel = walk_forward_panel()
    candidate = TargetCandidate(
        target_id="custom_20d",
        display_name="Custom 20d",
        label_column="label_custom_20d",
        target_type="test",
        horizon=20,
        description="Test target.",
        positive_label_meaning="Test positive.",
    )

    comparison = build_target_walk_forward_comparison(
        panel,
        ["feature_signal"],
        [candidate],
        train_window=20,
        test_window=10,
        step=10,
        embargo=20,
        min_sample_count=20,
        min_bucket_count=2,
    )

    row = comparison.iloc[0]
    assert row["target_id"] == "custom_20d"
    assert row["prediction_count"] > 0
    assert row["quality_summary"] in {"Promising", "Mixed", "Weak"}


def test_extreme_class_balance_is_flagged_without_crashing_walk_forward() -> None:
    panel = walk_forward_panel()
    panel["label_custom_20d"] = 1.0
    candidate = TargetCandidate(
        target_id="constant_20d",
        display_name="Constant 20d",
        label_column="label_custom_20d",
        target_type="test",
        horizon=20,
        description="Constant target.",
        positive_label_meaning="Always positive.",
    )

    comparison = build_target_walk_forward_comparison(
        panel,
        ["feature_signal"],
        [candidate],
        train_window=20,
        test_window=10,
        step=10,
        embargo=20,
        min_sample_count=20,
    )

    row = comparison.iloc[0]
    assert row["quality_summary"] == "Unusable"
    assert row["prediction_count"] == 0


def test_target_feature_group_comparison_represents_available_feature_groups() -> None:
    panel = walk_forward_panel()
    candidate = TargetCandidate(
        target_id="custom_20d",
        display_name="Custom 20d",
        label_column="label_custom_20d",
        target_type="test",
        horizon=20,
        description="Test target.",
        positive_label_meaning="Test positive.",
    )
    feature_groups = {
        "technical": ["feature_signal"],
        "technical_fourier": ["feature_signal", "feature_fourier"],
        "technical_wavelet": ["feature_signal", "feature_wavelet"],
        "all": ["feature_signal", "feature_fourier", "feature_wavelet"],
    }

    comparison = build_target_feature_group_comparison(
        panel,
        feature_groups,
        [candidate],
        train_window=20,
        test_window=10,
        step=10,
        embargo=20,
        min_sample_count=20,
        min_bucket_count=2,
    )

    assert set(comparison["feature_group"]) == {"technical", "technical_fourier", "technical_wavelet", "all"}
    assert (comparison["target_id"] == "custom_20d").all()
    assert (comparison["prediction_count"] > 0).all()


def test_target_regime_comparison_handles_single_class_regimes_without_crashing() -> None:
    panel = walk_forward_panel()
    panel["regime"] = np.where(panel["label_custom_20d"] == 1, "Positive-only regime", "Negative-only regime")
    candidate = TargetCandidate(
        target_id="custom_20d",
        display_name="Custom 20d",
        label_column="label_custom_20d",
        target_type="test",
        horizon=20,
        description="Test target.",
        positive_label_meaning="Test positive.",
    )

    comparison = build_target_regime_comparison(
        panel,
        ["feature_signal"],
        [candidate],
        train_window=20,
        test_window=10,
        step=10,
        embargo=20,
        min_target_sample_count=20,
        min_sample_count=2,
        min_bucket_count=2,
    )

    assert set(comparison["regime"]) == {"Positive-only regime", "Negative-only regime"}
    assert comparison["roc_auc"].isna().all()
    assert comparison["interpretation"].str.contains("one class").all()


def test_target_stability_summary_selects_best_feature_group_deterministically() -> None:
    candidate = TargetCandidate(
        target_id="custom_20d",
        display_name="Custom 20d",
        label_column="label_custom_20d",
        target_type="test",
        horizon=20,
        description="Test target.",
        positive_label_meaning="Test positive.",
    )
    feature_group_comparison = pd.DataFrame(
        [
            {
                "target_id": "custom_20d",
                "display_name": "Custom 20d",
                "feature_group": "technical",
                "prediction_count": 20,
                "positive_rate": 0.50,
                "roc_auc": 0.54,
                "pr_auc": 0.55,
                "bucket_spread": 0.04,
                "quality_summary": "Mixed",
            },
            {
                "target_id": "custom_20d",
                "display_name": "Custom 20d",
                "feature_group": "technical_wavelet",
                "prediction_count": 20,
                "positive_rate": 0.50,
                "roc_auc": 0.57,
                "pr_auc": 0.58,
                "bucket_spread": 0.12,
                "quality_summary": "Promising",
            },
        ]
    )
    regime_comparison = pd.DataFrame(
        [
            {"target_id": "custom_20d", "direction": "positive", "bucket_spread": 0.10},
            {"target_id": "custom_20d", "direction": "flat", "bucket_spread": 0.02},
        ]
    )

    summary = build_target_stability_summary(feature_group_comparison, regime_comparison, [candidate])
    row = summary.iloc[0]

    assert row["best_feature_group"] == "technical_wavelet"
    assert row["best_feature_group_metric"] == pytest.approx(0.12)
    assert row["overall_stability"] in {"Strong candidate", "Promising but regime-sensitive", "Feature-group dependent"}


def test_target_stability_summary_flags_inverted_regime_results() -> None:
    candidate = TargetCandidate(
        target_id="custom_20d",
        display_name="Custom 20d",
        label_column="label_custom_20d",
        target_type="test",
        horizon=20,
        description="Test target.",
        positive_label_meaning="Test positive.",
    )
    feature_group_comparison = pd.DataFrame(
        [
            {
                "target_id": "custom_20d",
                "display_name": "Custom 20d",
                "feature_group": "technical",
                "prediction_count": 20,
                "positive_rate": 0.50,
                "roc_auc": 0.60,
                "pr_auc": 0.62,
                "bucket_spread": 0.15,
                "quality_summary": "Promising",
            }
        ]
    )
    regime_comparison = pd.DataFrame(
        [
            {"target_id": "custom_20d", "direction": "positive", "bucket_spread": 0.12},
            {"target_id": "custom_20d", "direction": "inverted", "bucket_spread": -0.08},
        ]
    )

    summary = build_target_stability_summary(feature_group_comparison, regime_comparison, [candidate])
    row = summary.iloc[0]

    assert row["regime_negative_count"] == 1
    assert row["worst_regime_bucket_spread"] == pytest.approx(-0.08)
    assert row["overall_stability"] == "Promising but regime-sensitive"


def target_quality_inputs(
    *,
    target_id: str = "alternative_20d",
    display_name: str = "Alternative 20d",
    sample_count: int = 240,
    positive_rate: float = 0.48,
    class_balance_status: str = "Healthy",
    roc_auc: float = 0.62,
    pr_auc: float = 0.66,
    brier_score: float = 0.18,
    calibration_gap: float = 0.04,
    bucket_spread: float = 0.14,
    feature_rows: list[dict[str, object]] | None = None,
    regime_rows: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balance = pd.DataFrame(
        [
            {
                "target_id": target_id,
                "display_name": display_name,
                "sample_count": sample_count,
                "positive_rate": positive_rate,
                "class_balance_status": class_balance_status,
            }
        ]
    )
    walk_forward = pd.DataFrame(
        [
            {
                "target_id": target_id,
                "display_name": display_name,
                "prediction_count": sample_count // 2,
                "positive_rate": positive_rate,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier_score,
                "calibration_gap": calibration_gap,
                "bucket_spread": bucket_spread,
            }
        ]
    )
    feature_group = pd.DataFrame(
        feature_rows
        if feature_rows is not None
        else [
            {
                "target_id": target_id,
                "display_name": display_name,
                "feature_group": "technical",
                "prediction_count": 80,
                "positive_rate": positive_rate,
                "roc_auc": 0.61,
                "pr_auc": 0.64,
                "bucket_spread": 0.12,
                "quality_summary": "Promising",
            },
            {
                "target_id": target_id,
                "display_name": display_name,
                "feature_group": "all",
                "prediction_count": 80,
                "positive_rate": positive_rate,
                "roc_auc": 0.63,
                "pr_auc": 0.66,
                "bucket_spread": 0.15,
                "quality_summary": "Promising",
            },
        ]
    )
    regime = pd.DataFrame(
        regime_rows
        if regime_rows is not None
        else [
            {"target_id": target_id, "direction": "positive", "bucket_spread": 0.11},
            {"target_id": target_id, "direction": "positive", "bucket_spread": 0.08},
        ]
    )
    return balance, walk_forward, feature_group, regime


def test_target_quality_summary_marks_strong_alternative_as_production_trial_candidate() -> None:
    summary = build_target_quality_summary(*target_quality_inputs())
    row = summary.set_index("target_id").loc["alternative_20d"]

    assert row["overall_target_quality"] == "Promising"
    assert row["production_candidate_status"] == "Candidate for production trial"


def test_target_quality_summary_keeps_baseline_when_alternative_does_not_clearly_beat_it() -> None:
    baseline = target_quality_inputs(
        target_id="outperform_20d",
        display_name="Current 20d outperformance",
        roc_auc=0.59,
        bucket_spread=0.11,
    )
    alternative = target_quality_inputs(
        target_id="alternative_20d",
        display_name="Alternative 20d",
        roc_auc=0.60,
        bucket_spread=0.12,
    )
    summary = build_target_quality_summary(
        pd.concat([baseline[0], alternative[0]], ignore_index=True),
        pd.concat([baseline[1], alternative[1]], ignore_index=True),
        pd.concat([baseline[2], alternative[2]], ignore_index=True),
        pd.concat([baseline[3], alternative[3]], ignore_index=True),
    ).set_index("target_id")

    assert summary.loc["outperform_20d", "production_candidate_status"] == "Keep baseline"
    assert summary.loc["alternative_20d", "production_candidate_status"] == "Research-only candidate"
    assert "current production target" in summary.loc["outperform_20d", "recommended_next_step"]


def test_target_quality_summary_flags_good_overall_but_regime_sensitive_target() -> None:
    summary = build_target_quality_summary(
        *target_quality_inputs(
            regime_rows=[
                {"target_id": "alternative_20d", "direction": "positive", "bucket_spread": 0.14},
                {"target_id": "alternative_20d", "direction": "inverted", "bucket_spread": -0.07},
            ]
        )
    )
    row = summary.set_index("target_id").loc["alternative_20d"]

    assert row["regime_stability"] in {"Regime-sensitive", "Inverted in some regimes"}
    assert row["production_candidate_status"] != "Candidate for production trial"


def test_target_quality_summary_blocks_promising_label_when_calibration_is_poor() -> None:
    summary = build_target_quality_summary(
        *target_quality_inputs(calibration_gap=0.18, brier_score=0.32)
    )
    row = summary.set_index("target_id").loc["alternative_20d"]

    assert row["calibration_quality"] == "Poor"
    assert row["overall_target_quality"] != "Promising"


def test_target_quality_summary_marks_skewed_pullback_as_special_setup_only() -> None:
    summary = build_target_quality_summary(
        *target_quality_inputs(
            target_id="pullback_recovery_20d",
            display_name="Pullback and recovery",
            positive_rate=0.12,
            class_balance_status="Skewed",
            pr_auc=0.24,
        )
    )
    row = summary.set_index("target_id").loc["pullback_recovery_20d"]

    assert row["production_candidate_status"] == "Special setup only"
    assert row["production_candidate_status"] != "Candidate for production trial"


def test_target_quality_summary_marks_unusable_class_balance() -> None:
    summary = build_target_quality_summary(
        *target_quality_inputs(positive_rate=0.03, class_balance_status="Unusable")
    )
    row = summary.set_index("target_id").loc["alternative_20d"]

    assert row["overall_target_quality"] == "Unusable"


def test_target_quality_summary_detects_feature_group_dependent_target() -> None:
    feature_rows = [
        {
            "target_id": "alternative_20d",
            "display_name": "Alternative 20d",
            "feature_group": "technical",
            "prediction_count": 80,
            "positive_rate": 0.50,
            "roc_auc": 0.63,
            "pr_auc": 0.66,
            "bucket_spread": 0.14,
            "quality_summary": "Promising",
        },
        {
            "target_id": "alternative_20d",
            "display_name": "Alternative 20d",
            "feature_group": "technical_wavelet",
            "prediction_count": 80,
            "positive_rate": 0.50,
            "roc_auc": 0.49,
            "pr_auc": 0.48,
            "bucket_spread": -0.09,
            "quality_summary": "Weak",
        },
    ]
    summary = build_target_quality_summary(*target_quality_inputs(feature_rows=feature_rows))
    row = summary.set_index("target_id").loc["alternative_20d"]

    assert row["feature_group_consistency"] == "Feature-group dependent"


def test_target_quality_summary_ranking_is_deterministic() -> None:
    strong = target_quality_inputs(target_id="strong_20d", display_name="Strong 20d")
    weak = target_quality_inputs(
        target_id="weak_20d",
        display_name="Weak 20d",
        roc_auc=0.49,
        pr_auc=0.47,
        bucket_spread=-0.08,
        feature_rows=[
            {
                "target_id": "weak_20d",
                "display_name": "Weak 20d",
                "feature_group": "technical",
                "prediction_count": 80,
                "positive_rate": 0.48,
                "roc_auc": 0.49,
                "pr_auc": 0.47,
                "bucket_spread": -0.08,
                "quality_summary": "Weak",
            }
        ],
        regime_rows=[{"target_id": "weak_20d", "direction": "inverted", "bucket_spread": -0.08}],
    )
    args = (
        pd.concat([strong[0], weak[0]], ignore_index=True),
        pd.concat([strong[1], weak[1]], ignore_index=True),
        pd.concat([strong[2], weak[2]], ignore_index=True),
        pd.concat([strong[3], weak[3]], ignore_index=True),
    )

    first = build_target_quality_summary(*args)[["target_id", "candidate_rank"]]
    second = build_target_quality_summary(*args)[["target_id", "candidate_rank"]]

    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["target_id"] == "strong_20d"


def test_target_arena_comparison_exports_requested_research_candidates() -> None:
    baseline = target_quality_inputs(
        target_id="outperform_20d",
        display_name="Current 20d outperformance",
        roc_auc=0.52,
        pr_auc=0.48,
        brier_score=0.26,
        calibration_gap=0.03,
        bucket_spread=0.04,
        regime_rows=[
            {"target_id": "outperform_20d", "direction": "positive", "bucket_spread": 0.09},
            {"target_id": "outperform_20d", "direction": "flat", "bucket_spread": 0.01},
        ],
    )
    top_tercile = target_quality_inputs(
        target_id="top_tercile_excess_20d",
        display_name="Top-third relative performer",
        positive_rate=0.34,
        roc_auc=0.57,
        pr_auc=0.42,
        brier_score=0.24,
        calibration_gap=0.04,
        bucket_spread=0.09,
    )
    risk_adjusted = target_quality_inputs(
        target_id="risk_adjusted_excess_20d",
        display_name="Recent-vol adjusted excess",
        roc_auc=0.48,
        bucket_spread=-0.04,
        regime_rows=[{"target_id": "risk_adjusted_excess_20d", "direction": "inverted", "bucket_spread": -0.07}],
    )
    drawdown_adjusted = target_quality_inputs(
        target_id="drawdown_adjusted_opportunity_20d",
        display_name="Drawdown-adjusted opportunity",
        roc_auc=0.54,
        pr_auc=0.40,
        brier_score=0.24,
        calibration_gap=0.04,
        bucket_spread=0.06,
    )
    pullback = target_quality_inputs(
        target_id="pullback_recovery_20d",
        display_name="Pullback and recovery",
        positive_rate=0.12,
        class_balance_status="Skewed",
        roc_auc=0.55,
        pr_auc=0.20,
        brier_score=0.16,
        calibration_gap=0.08,
        bucket_spread=0.03,
    )
    summary = build_target_quality_summary(
        pd.concat([baseline[0], top_tercile[0], risk_adjusted[0], drawdown_adjusted[0], pullback[0]], ignore_index=True),
        pd.concat([baseline[1], top_tercile[1], risk_adjusted[1], drawdown_adjusted[1], pullback[1]], ignore_index=True),
        pd.concat([baseline[2], top_tercile[2], risk_adjusted[2], drawdown_adjusted[2], pullback[2]], ignore_index=True),
        pd.concat([baseline[3], top_tercile[3], risk_adjusted[3], drawdown_adjusted[3], pullback[3]], ignore_index=True),
    )

    arena = build_target_arena_comparison(summary).set_index("target_id")

    assert arena.index.tolist() == [
        "outperform_20d",
        "top_tercile_excess_20d",
        "risk_adjusted_excess_20d",
        "drawdown_adjusted_opportunity_20d",
        "pullback_recovery_20d",
    ]
    assert set(arena["evidence_classification"]).issubset({"strong", "promising", "mixed", "weak", "insufficient"})
    assert bool(arena.loc["top_tercile_excess_20d", "future_production_experiment"]) is True
    assert arena.loc["risk_adjusted_excess_20d", "evidence_classification"] == "weak"
    assert "class balance" in arena.loc["pullback_recovery_20d", "rejection_reason"]


def test_production_ml_score_formula_remains_outperformance_probability_scaled() -> None:
    score = ml_score(pd.Series([0.25, 0.75]), pd.Series([0.90, 0.10]))

    pd.testing.assert_series_equal(score, pd.Series([25.0, 75.0]))
