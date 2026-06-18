from __future__ import annotations

import pandas as pd

from src.research.lab import build_opportunity_baseline_challenge


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
