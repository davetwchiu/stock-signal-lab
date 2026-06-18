from __future__ import annotations

import pandas as pd

from src.ml.validation import MLValidationResult
from src.research.lab import build_drawdown_risk_feature_group_incremental_value


def test_drawdown_risk_feature_group_incremental_value_uses_training_prevalence(monkeypatch) -> None:
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-10",
            "2024-01-10",
            "2024-01-11",
            "2024-01-11",
        ]
    )
    dataset = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "label_drawdown_risk_20d": [0, 0, 0, 1, 1, 1, 1, 0],
            "market_regime": ["risk_on"] * 8,
        }
    )
    predictions = pd.DataFrame(
        {
            "fold": [1, 1, 1, 1],
            "Date": dates[4:],
            "Ticker": ["A", "B", "A", "B"],
            "actual": [1, 1, 1, 0],
        }
    )
    fold_metrics = pd.DataFrame(
        {
            "fold": [1],
            "train_start": [pd.Timestamp("2024-01-01")],
            "train_end": [pd.Timestamp("2024-01-04")],
        }
    )

    def fake_walk_forward(dataset, columns, **kwargs):
        probability = [0.25, 0.25, 0.25, 0.25]
        if len(columns) > 1:
            probability = [0.9, 0.8, 0.7, 0.2]
        return MLValidationResult(
            predictions=predictions.assign(probability=probability),
            fold_metrics=fold_metrics,
            overall_metrics=pd.DataFrame(),
            fold_feature_importance=pd.DataFrame(),
        )

    monkeypatch.setattr("src.research.lab.walk_forward_validate_classifier", fake_walk_forward)

    table = build_drawdown_risk_feature_group_incremental_value(
        dataset,
        {"technical": ["a"], "all": ["a", "b"]},
        label_column="label_drawdown_risk_20d",
        model_name="current_default",
        train_window=4,
        test_window=4,
        step=4,
        embargo=0,
    )

    assert list(table["feature_group"]) == ["technical", "all"]
    assert table.loc[0, "fold_train_prevalence_details"] == "1:4:0.250000"
    assert table.loc[0, "event_prevalence"] == 0.75
    assert table.loc[1, "classification"] == "adds_incremental_risk_signal"
