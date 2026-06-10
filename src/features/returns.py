"""Return, volatility, and drawdown features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(price: pd.Series) -> pd.Series:
    """Daily percentage returns."""

    return price.pct_change()


def rolling_return(price: pd.Series, window: int) -> pd.Series:
    """Trailing return using only the current and prior prices."""

    return price / price.shift(window) - 1.0


def rolling_volatility(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Trailing return volatility."""

    vol = returns.rolling(window=window, min_periods=window).std()
    return vol * np.sqrt(252) if annualize else vol


def _window_max_drawdown(values: np.ndarray) -> float:
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    return float(np.nanmin(drawdowns))


def rolling_max_drawdown(price: pd.Series, window: int) -> pd.Series:
    """Worst drawdown observed inside each trailing window."""

    return price.rolling(window=window, min_periods=window).apply(_window_max_drawdown, raw=True)


def add_return_features(
    data: pd.DataFrame,
    return_windows: tuple[int, ...] = (5, 20, 60, 120),
    volatility_windows: tuple[int, ...] = (20, 60),
    drawdown_windows: tuple[int, ...] = (60, 120),
) -> pd.DataFrame:
    """Add return, volatility, and drawdown features to an OHLCV frame."""

    output = data.copy()
    price = output["Adj Close"]
    output["daily_return"] = daily_returns(price)

    for window in return_windows:
        output[f"return_{window}d"] = rolling_return(price, window)
    for window in volatility_windows:
        output[f"volatility_{window}d"] = rolling_volatility(output["daily_return"], window)
    for window in drawdown_windows:
        output[f"max_drawdown_{window}d"] = rolling_max_drawdown(price, window)

    return output

