"""Technical and relative-strength features."""

from __future__ import annotations

import pandas as pd

from src.features.returns import add_return_features, rolling_return
from src.utils.config import FeatureConfig


def distance_from_moving_average(price: pd.Series, window: int) -> pd.Series:
    """Percentage distance from a trailing moving average."""

    ma = price.rolling(window=window, min_periods=window).mean()
    return price / ma - 1.0


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """Trailing volume z-score."""

    mean = volume.rolling(window=window, min_periods=window).mean()
    std = volume.rolling(window=window, min_periods=window).std()
    return (volume - mean) / std.replace(0, pd.NA)


def rsi_momentum(price: pd.Series, window: int = 14) -> pd.Series:
    """RSI-like momentum oscillator implemented with simple rolling averages."""

    delta = price.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def add_relative_strength(
    features: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame] | None = None,
    windows: tuple[int, ...] = (60, 120),
) -> pd.DataFrame:
    """Add trailing relative-strength features versus benchmark prices."""

    output = features.copy()
    if not benchmark_frames:
        return output

    price = output["Adj Close"]
    for benchmark, frame in benchmark_frames.items():
        if frame.empty or "Adj Close" not in frame:
            continue
        benchmark_price = frame["Adj Close"].reindex(output.index).ffill()
        for window in windows:
            output[f"rs_{benchmark.lower()}_{window}d"] = (
                rolling_return(price, window) - rolling_return(benchmark_price, window)
            )
    return output


def build_technical_features(
    data: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame] | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build the MVP technical feature matrix for one ticker."""

    cfg = config or FeatureConfig()
    output = add_return_features(
        data,
        return_windows=cfg.return_windows,
        volatility_windows=cfg.volatility_windows,
        drawdown_windows=cfg.drawdown_windows,
    )
    price = output["Adj Close"]
    for window in cfg.moving_average_windows:
        output[f"ma_{window}d"] = price.rolling(window=window, min_periods=window).mean()
        output[f"dist_ma_{window}d"] = distance_from_moving_average(price, window)
    for window in cfg.volume_z_windows:
        output[f"volume_z_{window}d"] = volume_zscore(output["Volume"], window)
    output[f"rsi_{cfg.rsi_window}d"] = rsi_momentum(price, cfg.rsi_window)
    return add_relative_strength(output, benchmark_frames=benchmark_frames)

