"""Small CSV cache for downloaded daily OHLCV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import CACHE_DIR


def normalize_ticker(ticker: str) -> str:
    """Return a filesystem-safe ticker symbol."""

    return ticker.strip().upper().replace("/", "_").replace(" ", "")


def cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Return the local cache path for a ticker."""

    return cache_dir / f"{normalize_ticker(ticker)}.csv"


def read_cached(ticker: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame | None:
    """Read cached data if it exists."""

    path = cache_path(ticker, cache_dir)
    if not path.exists():
        return None
    data = pd.read_csv(path, parse_dates=["Date"])
    return data.set_index("Date").sort_index()


def write_cache(ticker: str, data: pd.DataFrame, cache_dir: Path = CACHE_DIR) -> Path:
    """Persist daily OHLCV data to the local cache."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(ticker, cache_dir)
    output = data.copy()
    if output.index.name != "Date":
        output.index.name = "Date"
    output.reset_index().to_csv(path, index=False)
    return path

