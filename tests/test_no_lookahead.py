from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.features.fourier import rolling_fourier_features
from src.features.technical import build_technical_features
from src.features.wavelet import rolling_wavelet_features


def synthetic_prices(rows: int = 180) -> pd.Series:
    index = pd.date_range("2021-01-01", periods=rows, freq="B")
    trend = np.linspace(100, 140, rows)
    cycle = 2 * np.sin(np.arange(rows) / 5)
    return pd.Series(trend + cycle, index=index)


def ohlcv_from_price(price: pd.Series) -> pd.DataFrame:
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


def test_rolling_features_do_not_change_when_future_prices_change() -> None:
    price = synthetic_prices()
    mutated = price.copy()
    cutoff = price.index[100]
    mutated.loc[mutated.index > cutoff] *= 3.0

    original_features = build_technical_features(ohlcv_from_price(price))
    mutated_features = build_technical_features(ohlcv_from_price(mutated))

    pd.testing.assert_series_equal(
        original_features.loc[:cutoff, "return_60d"],
        mutated_features.loc[:cutoff, "return_60d"],
    )
    pd.testing.assert_series_equal(
        original_features.loc[:cutoff, "max_drawdown_60d"],
        mutated_features.loc[:cutoff, "max_drawdown_60d"],
    )


def test_fourier_features_do_not_change_when_future_prices_change() -> None:
    price = synthetic_prices()
    mutated = price.copy()
    cutoff = price.index[100]
    mutated.loc[mutated.index > cutoff] *= 3.0

    original = rolling_fourier_features(price, window=40, n_components=2)
    changed = rolling_fourier_features(mutated, window=40, n_components=2)

    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_wavelet_features_do_not_change_when_future_prices_change() -> None:
    price = synthetic_prices()
    mutated = price.copy()
    cutoff = price.index[100]
    mutated.loc[mutated.index > cutoff] *= 3.0

    original = rolling_wavelet_features(price, window=64, level=2)
    changed = rolling_wavelet_features(mutated, window=64, level=2)

    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_backtest_does_not_use_same_day_signal_for_same_day_return() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    price = pd.Series([100.0, 50.0, 100.0], index=index)
    signal = pd.Series([0.0, 1.0, 0.0], index=index)

    result = run_backtest(price, signal, transaction_cost_bps=0.0, slippage_bps=0.0)

    assert result.curve["strategy_return"].iloc[1] == 0.0
    assert result.curve["strategy_return"].iloc[2] == 1.0

