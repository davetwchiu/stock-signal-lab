"""Configuration defaults for the MVP research app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

DEFAULT_TICKERS: tuple[str, ...] = (
    "NVDA",
    "TSM",
    "AMD",
    "AVGO",
    "MU",
    "LITE",
    "COHR",
    "CRDO",
    "XLK",
    "SMH",
    "SOXX",
    "QQQ",
    "SPY",
)

CSV_COLUMNS: tuple[str, ...] = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature window defaults used by both scripts and the UI."""

    return_windows: tuple[int, ...] = (5, 20, 60, 120)
    volatility_windows: tuple[int, ...] = (20, 60)
    drawdown_windows: tuple[int, ...] = (60, 120)
    moving_average_windows: tuple[int, ...] = (50, 200)
    volume_z_windows: tuple[int, ...] = (20, 60)
    rsi_window: int = 14
    fourier_window: int = 60
    fourier_components: int = 3
    fourier_input: str = "returns"
    wavelet_window: int = 64
    wavelet: str = "db4"
    wavelet_level: int = 3


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest assumptions that should be visible and easy to change."""

    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_capital: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    """Data-provider settings for local-first usage."""

    cache_dir: Path = CACHE_DIR
    default_tickers: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TICKERS)

