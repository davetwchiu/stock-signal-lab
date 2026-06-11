"""Daily market data loading.

The provider protocol keeps feature and backtest code independent from the
source of OHLCV data. `yfinance` is the MVP provider; paid providers can be
added later by implementing the same `fetch_daily` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.data.cache import read_cached, write_cache
from src.utils.config import CACHE_DIR, CSV_COLUMNS


class MarketDataProvider(Protocol):
    """Interface for daily OHLCV providers."""

    def fetch_daily(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data indexed by date."""


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily OHLCV columns and index."""

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    frame = data.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date")
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"

    missing = [column for column in CSV_COLUMNS[1:] if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV data missing required columns: {missing}")

    frame = frame.loc[:, list(CSV_COLUMNS[1:])]
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.dropna(subset=["Adj Close"])


@dataclass
class CSVDataProvider:
    """Provider for local CSV files."""

    path: Path

    def fetch_daily(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        del ticker
        data = normalize_ohlcv(pd.read_csv(self.path, parse_dates=["Date"]))
        return filter_dates(data, start, end)


@dataclass
class YFinanceProvider:
    """Public-data MVP provider backed by yfinance."""

    auto_adjust: bool = False

    def fetch_daily(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Install yfinance to download market data.") from exc

        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=False,
        )
        if data.empty:
            raise ValueError(f"No data returned for {ticker}.")
        return normalize_ohlcv(data)


def filter_dates(
    data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Filter a date-indexed frame by optional inclusive dates."""

    output = data
    if start:
        output = output.loc[output.index >= pd.Timestamp(start)]
    if end:
        output = output.loc[output.index <= pd.Timestamp(end)]
    return output.copy()


def cache_satisfies(
    data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> bool:
    """Return whether a cached frame appears to cover the requested range."""

    if data.empty:
        return False
    if start and data.index.min() > pd.Timestamp(start):
        return False
    if end and data.index.max() < pd.Timestamp(end):
        return False
    return True


def load_daily_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    provider: MarketDataProvider | None = None,
    csv_path: str | Path | None = None,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Load daily OHLCV data from CSV, cache, or the configured provider."""

    if csv_path:
        return CSVDataProvider(Path(csv_path)).fetch_daily(ticker, start, end)

    if use_cache:
        cached = read_cached(ticker, cache_dir)
        if cached is not None and cache_satisfies(cached, start, end):
            return filter_dates(cached, start, end)

    active_provider = provider or YFinanceProvider()
    data = active_provider.fetch_daily(ticker, start, end)
    if use_cache:
        write_cache(ticker, data, cache_dir)
    return filter_dates(data, start, end)


def load_universe(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    provider: MarketDataProvider | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load several tickers and skip symbols that fail provider lookup."""

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        clean = ticker.strip().upper()
        if not clean:
            continue
        frames[clean] = load_daily_data(
            clean,
            start=start,
            end=end,
            provider=provider,
            use_cache=use_cache,
        )
    return frames

