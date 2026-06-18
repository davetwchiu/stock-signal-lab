from __future__ import annotations

import pandas as pd

from src.ml.validation import MLValidationResult
from src.research.lab import (
    build_adverse_outcome_label_comparison,
    build_drawdown_risk_feature_group_incremental_value,
)


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


def test_adverse_outcome_label_comparison_uses_fixed_research_labels(monkeypatch) -> None:
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
            "forward_20d_drawdown": [-0.01, -0.02, -0.20, -0.03, -0.18, -0.04, -0.19, -0.01],
            "forward_20d_excess_return": [0.01, -0.01, -0.06, 0.02, -0.07, -0.01, -0.08, 0.01],
            "forward_20d_return": [0.02, 0.01, -0.10, 0.03, -0.11, 0.01, -0.12, 0.02],
            "volatility_20d": [0.20] * 8,
            "label_drawdown_risk_20d": [0, 0, 1, 0, 1, 0, 1, 0],
            "market_regime": [
                "risk_on",
                "risk_on",
                "risk_off",
                "risk_on",
                "risk_off",
                "risk_on",
                "risk_off",
                "risk_on",
            ],
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    predictions = pd.DataFrame(
        {
            "fold": [1, 1, 1, 1],
            "Date": dates[4:],
            "Ticker": ["A", "B", "A", "B"],
            "actual": [1, 0, 1, 0],
            "probability": [0.9, 0.2, 0.8, 0.1],
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
        assert kwargs["label_column"] in {
            "risk_severe_drawdown_20d",
            "risk_negative_excess_20d",
            "risk_composite_drawdown_underperform_20d",
            "risk_vol_adjusted_adverse_20d",
        }
        return MLValidationResult(
            predictions=predictions.assign(
                actual=dataset.loc[4:, kwargs["label_column"]].astype(int).to_numpy()
            ),
            fold_metrics=fold_metrics,
            overall_metrics=pd.DataFrame(),
            fold_feature_importance=pd.DataFrame(),
        )

    monkeypatch.setattr("src.research.lab.walk_forward_validate_classifier", fake_walk_forward)

    table = build_adverse_outcome_label_comparison(
        dataset,
        ["feature"],
        current_label_column="label_drawdown_risk_20d",
        horizon=20,
        model_name="current_default",
        train_window=4,
        test_window=4,
        step=4,
        embargo=0,
    )

    assert set(table["label"]) == {
        "risk_severe_drawdown_20d",
        "risk_negative_excess_20d",
        "risk_composite_drawdown_underperform_20d",
        "risk_vol_adjusted_adverse_20d",
    }
    severe = table.set_index("label").loc["risk_severe_drawdown_20d"]
    assert severe["threshold"] == "forward_drawdown < -0.15"
    assert severe["event_prevalence"] == 0.5
    assert severe["current_label_overlap_rate"] == 1.0
    assert severe["model_vs_global_fold_baseline"] == "model_beats_baseline"
    assert severe["classification"] == "candidate_beats_baseline"
