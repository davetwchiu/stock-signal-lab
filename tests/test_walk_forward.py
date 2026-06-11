from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.validation import date_walk_forward_splits, walk_forward_validate_classifier


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
    for fold in result.predictions["fold"].unique():
        fold_predictions = result.predictions[result.predictions["fold"] == fold]
        metric_row = result.fold_metrics[result.fold_metrics["fold"] == fold].iloc[0]
        assert fold_predictions["Date"].min() == metric_row["test_start"]

