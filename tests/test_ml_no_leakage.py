from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fourier import rolling_fourier_features
from src.features.technical import build_technical_features
from src.ml.datasets import assert_no_label_leakage, build_supervised_panel, feature_group_columns


def ohlcv(price: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": price,
            "High": price + 1,
            "Low": price - 1,
            "Close": price,
            "Adj Close": price,
            "Volume": 1_000_000,
        },
        index=price.index,
    )


def test_labels_are_not_feature_columns() -> None:
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    price = pd.Series(np.linspace(100, 140, len(index)), index=index)
    features = build_technical_features(ohlcv(price)).join(rolling_fourier_features(price, window=40))
    panel = build_supervised_panel({"AAA": features}, benchmark_price=price, horizon=20)

    columns = feature_group_columns(panel, "all")

    assert columns
    assert all(not column.startswith(("label_", "forward_", "benchmark_forward_")) for column in columns)
    with pytest.raises(ValueError):
        assert_no_label_leakage(columns + ["label_outperform_20d"])


def test_future_price_mutation_changes_labels_not_historical_features() -> None:
    index = pd.date_range("2020-01-01", periods=160, freq="B")
    price = pd.Series(np.linspace(100, 140, len(index)), index=index)
    mutated = price.copy()
    cutoff = index[80]
    mutated.loc[mutated.index > cutoff] *= 2.0

    features = build_technical_features(ohlcv(price))
    changed_features = build_technical_features(ohlcv(mutated))
    benchmark = pd.Series(100.0, index=index)
    labels = build_supervised_panel({"AAA": features}, benchmark, horizon=20)
    changed_labels = build_supervised_panel({"AAA": changed_features}, benchmark, horizon=20)

    pd.testing.assert_series_equal(features.loc[:cutoff, "return_60d"], changed_features.loc[:cutoff, "return_60d"])
    original_label = labels.loc[labels["Date"] == cutoff, "forward_20d_return"].iloc[0]
    changed_label = changed_labels.loc[changed_labels["Date"] == cutoff, "forward_20d_return"].iloc[0]
    assert changed_label != original_label

