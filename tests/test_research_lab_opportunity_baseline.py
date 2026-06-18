from __future__ import annotations

import pandas as pd

from src.research.lab import (
    build_opportunity_baseline_challenge,
    build_opportunity_label_baseline_challenge,
    build_opportunity_label_baseline_tables,
)


def test_opportunity_baseline_challenge_uses_training_fold_prevalence() -> None:
    train_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    test_dates = pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13"])
    panel = pd.DataFrame(
        {
            "Date": [*train_dates, *test_dates],
            "Ticker": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "label_outperform_20d": [0, 0, 0, 1, 1, 0, 0, 0],
            "market_regime": ["calm", "calm", "risk_on", "risk_on", "calm", "calm", "risk_on", "risk_on"],
            "momentum_60d": [1.0, 2.0, 3.0, 4.0, 4.0, 1.0, 2.0, 3.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "fold": [1, 1, 1, 1],
            "Date": test_dates,
            "Ticker": ["A", "B", "C", "D"],
            "actual": [1, 0, 0, 0],
            "probability": [0.1, 0.9, 0.9, 0.9],
        }
    )
    folds = pd.DataFrame(
        {
            "fold": [1],
            "train_start": [train_dates.min()],
            "train_end": [train_dates.max()],
        }
    )

    table = build_opportunity_baseline_challenge(
        predictions,
        panel,
        folds,
        bucket_count=2,
        min_train_samples=1,
        min_train_events=0,
    ).set_index("comparator")

    assert table.loc["global_fold_prevalence_baseline", "mean_predicted_opportunity"] == 0.25
    assert table.loc["global_fold_prevalence_baseline", "fold_train_prevalence_details"] == "1:4:0.250000"
    assert table.loc["momentum_bucket_prevalence_baseline", "momentum_feature"] == "momentum_60d"
    assert table.loc["momentum_bucket_prevalence_baseline", "bucket_count"] == 2
    assert table.loc["model_predicted_opportunity", "classification"] == "baseline_beats_model"
    assert set(table.index) == {
        "model_predicted_opportunity",
        "global_fold_prevalence_baseline",
        "regime_fold_prevalence_baseline",
        "momentum_bucket_prevalence_baseline",
        "regime_momentum_bucket_prevalence_baseline",
    }


def test_opportunity_label_baseline_challenge_compares_fixed_candidate_labels() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        for ticker, excess, drawdown, signal in (
            ("AAA", 0.08, -0.03, 0.9),
            ("BBB", 0.03, -0.12, 0.5),
            ("CCC", -0.01, -0.02, 0.1),
        ):
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "signal": signal,
                    "forward_20d_excess_return": excess,
                    "forward_20d_drawdown": drawdown,
                    "label_outperform_20d": float(excess > 0.02),
                    "label_top_tercile_excess_20d": float(ticker == "AAA"),
                    "label_risk_adjusted_excess_20d": float(excess > 0.0),
                    "market_regime": "calm" if ticker != "BBB" else "risk_on",
                    "rs_qqq_60d": signal,
                }
            )
    panel = pd.DataFrame(rows)

    table = build_opportunity_label_baseline_challenge(
        panel,
        ["signal"],
        horizon=20,
        model_name="logistic_regression",
        train_window=5,
        test_window=5,
        step=5,
        embargo=0,
        min_train_samples=1,
        min_train_events=0,
    )

    assert set(table["target_id"]) == {
        "outperform_20d",
        "stronger_excess_20d",
        "top_tercile_excess_20d",
        "risk_adjusted_excess_20d",
        "composite_opportunity_20d",
    }
    assert set(table["comparator"]) == {
        "model_predicted_opportunity",
        "global_fold_prevalence_baseline",
        "regime_fold_prevalence_baseline",
        "momentum_bucket_prevalence_baseline",
        "regime_momentum_bucket_prevalence_baseline",
    }
    composite = table[table["target_id"] == "composite_opportunity_20d"]
    assert composite["label_column"].iat[0] == "label_composite_opportunity_20d"
    assert composite["sample_count"].iat[0] > 0


def test_opportunity_label_baseline_tables_include_local_breakdowns() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    panel = pd.DataFrame(
        [
            {
                "Date": date,
                "Ticker": ticker,
                "signal": signal,
                "forward_20d_excess_return": excess,
                "forward_20d_drawdown": -0.02,
                "label_outperform_20d": float(excess > 0.02),
                "label_top_tercile_excess_20d": float(ticker == "AAA"),
                "label_risk_adjusted_excess_20d": float(excess > 0.0),
                "market_regime": "calm" if ticker != "BBB" else "risk_on",
                "rs_qqq_60d": signal,
            }
            for date in dates
            for ticker, excess, signal in (
                ("AAA", 0.08, 0.9),
                ("BBB", 0.03, 0.5),
                ("CCC", -0.01, 0.1),
            )
        ]
    )

    _, breakdown = build_opportunity_label_baseline_tables(
        panel,
        ["signal"],
        horizon=20,
        model_name="logistic_regression",
        train_window=5,
        test_window=5,
        step=5,
        embargo=0,
        min_train_samples=1,
        min_train_events=0,
    )

    risk_adjusted = breakdown[breakdown["target_id"] == "risk_adjusted_excess_20d"]
    assert set(risk_adjusted["breakdown"]) == {"fold", "ticker", "regime"}
    assert "global_fold_prevalence_baseline" in set(risk_adjusted["comparator"])
