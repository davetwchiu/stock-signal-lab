from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.validation import (
    date_walk_forward_splits,
    deduplicate_prediction_keys,
    prediction_merge_keys,
    walk_forward_validate_classifier,
)


class DirectionalCandidatePipeline:
    named_steps = {"model": type("DirectionalModel", (), {"classes_": np.array([0, 1])})()}

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, features: pd.DataFrame, label: pd.Series) -> "DirectionalCandidatePipeline":
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = features["signal"].astype(float).clip(0.01, 0.99)
        if self.name == "inverse_signal_model":
            probability = 1.0 - probability
        return np.column_stack([1.0 - probability, probability])


def synthetic_dataset() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=140, freq="B")
    rows = []
    for ticker_offset, ticker in enumerate(["AAA", "BBB"]):
        for idx, dt in enumerate(dates):
            value = idx + ticker_offset
            rows.append(
                {
                    "Date": dt,
                    "Ticker": ticker,
                    "return_20d": value / 100.0,
                    "volatility_20d": abs(np.sin(value)),
                    "label_outperform_20d": int(value % 7 > 3),
                    "forward_20d_return": value / 1000.0,
                    "forward_20d_excess_return": value / 2000.0,
                    "forward_20d_drawdown": -value / 5000.0,
                }
            )
    return pd.DataFrame(rows)


def selection_dataset(periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    rows = []
    for ticker in ["AAA", "BBB"]:
        for idx, dt in enumerate(dates):
            signal = 0.90 if idx % 4 in (0, 1) else 0.10
            rows.append(
                {
                    "Date": dt,
                    "Ticker": ticker,
                    "signal": signal,
                    "label_outperform_20d": int(signal > 0.5),
                    "label_drawdown_risk_20d": int(signal < 0.5),
                    "forward_20d_return": signal / 100.0,
                    "forward_20d_excess_return": signal / 100.0,
                    "forward_20d_drawdown": -signal / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_date_walk_forward_splits_respect_embargo() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=100, freq="B"))
    splits = date_walk_forward_splits(dates, train_window=40, test_window=10, step=10, embargo=5)

    train_dates, test_dates = splits[0]

    assert train_dates[-1] < test_dates[0]
    assert (dates[dates == test_dates[0]].index[0] - dates[dates == train_dates[-1]].index[0]) == 6


def test_walk_forward_classifier_outputs_future_fold_predictions() -> None:
    data = synthetic_dataset()
    result = walk_forward_validate_classifier(
        data,
        feature_columns=["return_20d", "volatility_20d"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=20,
        step=20,
        embargo=5,
    )

    assert not result.predictions.empty
    assert not result.fold_metrics.empty
    assert "selected_model" in result.fold_metrics
    for fold in result.predictions["fold"].unique():
        fold_predictions = result.predictions[result.predictions["fold"] == fold]
        metric_row = result.fold_metrics[result.fold_metrics["fold"] == fold].iloc[0]
        assert fold_predictions["Date"].min() == metric_row["test_start"]


def test_walk_forward_classifier_uses_label_horizon_as_minimum_embargo() -> None:
    data = synthetic_dataset()
    result = walk_forward_validate_classifier(
        data,
        feature_columns=["return_20d", "volatility_20d"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=10,
        step=10,
        embargo=0,
    )

    assert not result.fold_metrics.empty
    assert result.fold_metrics["requested_embargo"].eq(0).all()
    assert result.fold_metrics["effective_embargo"].eq(20).all()
    first = result.fold_metrics.iloc[0]
    dates = pd.Series(pd.date_range("2020-01-01", periods=140, freq="B"))
    gap = dates[dates == first["test_start"]].index[0] - dates[dates == first["train_end"]].index[0]
    assert gap == 21


def test_prediction_merge_helpers_prefer_fold_keys_and_deduplicate() -> None:
    left = pd.DataFrame(
        {
            "fold": [1, 2, 2],
            "Date": [pd.Timestamp("2024-01-02")] * 3,
            "Ticker": ["AAA", "AAA", "AAA"],
            "probability": [0.4, 0.5, 0.6],
        }
    )
    right = pd.DataFrame(
        {
            "fold": [1, 2],
            "Date": [pd.Timestamp("2024-01-02")] * 2,
            "Ticker": ["AAA", "AAA"],
            "probability": [0.3, 0.2],
        }
    )

    keys = prediction_merge_keys(left, right)
    deduplicated = deduplicate_prediction_keys(left, keys)

    assert keys == ["fold", "Date", "Ticker"]
    assert len(deduplicated) == 2
    assert deduplicated.loc[deduplicated["fold"] == 2, "probability"].iloc[0] == 0.6


def test_walk_forward_model_selection_is_deterministic() -> None:
    data = synthetic_dataset()

    first = walk_forward_validate_classifier(
        data,
        feature_columns=["return_20d", "volatility_20d"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=20,
        step=20,
        embargo=5,
        model_selection_mode="auto_select",
    )
    second = walk_forward_validate_classifier(
        data,
        feature_columns=["return_20d", "volatility_20d"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=20,
        step=20,
        embargo=5,
        model_selection_mode="auto_select",
    )

    assert first.fold_metrics["selected_model"].tolist() == second.fold_metrics["selected_model"].tolist()


def test_walk_forward_model_selection_uses_only_outer_train_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.ml.validation.available_model_candidates", lambda: ("signal_model", "inverse_signal_model"))
    monkeypatch.setattr(
        "src.ml.validation.build_classifier",
        lambda candidate_name, random_state=42: DirectionalCandidatePipeline(candidate_name),
    )
    data = selection_dataset(periods=105)
    base = walk_forward_validate_classifier(
        data,
        feature_columns=["signal"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=20,
        step=100,
        embargo=5,
        model_selection_mode="auto_select",
    )
    splits = date_walk_forward_splits(data["Date"], train_window=60, test_window=20, step=100, embargo=20)
    mutated = data.copy()
    mutated.loc[mutated["Date"].isin(splits[0][1]), "signal"] = 1.0 - mutated.loc[
        mutated["Date"].isin(splits[0][1]),
        "signal",
    ]
    mutated_result = walk_forward_validate_classifier(
        mutated,
        feature_columns=["signal"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=20,
        step=100,
        embargo=5,
        model_selection_mode="auto_select",
    )

    assert base.fold_metrics["selected_model"].tolist() == mutated_result.fold_metrics["selected_model"].tolist()


def test_walk_forward_selects_clear_inner_validation_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.ml.validation.available_model_candidates", lambda: ("inverse_signal_model", "signal_model"))
    monkeypatch.setattr(
        "src.ml.validation.build_classifier",
        lambda candidate_name, random_state=42: DirectionalCandidatePipeline(candidate_name),
    )

    result = walk_forward_validate_classifier(
        selection_dataset(),
        feature_columns=["signal"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=20,
        step=20,
        embargo=5,
        model_selection_mode="auto_select",
    )

    assert result.fold_metrics["selected_model"].eq("signal_model").all()
    assert result.fold_metrics["selection_metric"].eq("roc_auc").all()


def test_walk_forward_targets_can_select_different_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.ml.validation.available_model_candidates", lambda: ("signal_model", "inverse_signal_model"))
    monkeypatch.setattr(
        "src.ml.validation.build_classifier",
        lambda candidate_name, random_state=42: DirectionalCandidatePipeline(candidate_name),
    )
    data = selection_dataset()

    outperformance = walk_forward_validate_classifier(
        data,
        feature_columns=["signal"],
        label_column="label_outperform_20d",
        train_window=60,
        test_window=20,
        step=20,
        embargo=5,
        model_selection_mode="auto_select",
    )
    drawdown_risk = walk_forward_validate_classifier(
        data,
        feature_columns=["signal"],
        label_column="label_drawdown_risk_20d",
        train_window=60,
        test_window=20,
        step=20,
        embargo=5,
        model_selection_mode="auto_select",
    )

    assert outperformance.fold_metrics["selected_model"].eq("signal_model").all()
    assert drawdown_risk.fold_metrics["selected_model"].eq("inverse_signal_model").all()
    assert outperformance.fold_metrics["selected_model"].iloc[0] != drawdown_risk.fold_metrics["selected_model"].iloc[0]
