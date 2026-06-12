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


REQUIRED_OHLCV_COLUMNS = tuple(CSV_COLUMNS[1:])


def _canonical_ohlcv_label(label: object) -> str | None:
    """Return a canonical OHLCV label for common provider/cache variants."""

    text = str(label).strip()
    while "." in text and text.rsplit(".", maxsplit=1)[-1].isdigit():
        text = text.rsplit(".", maxsplit=1)[0].strip()

    normalized = " ".join(text.replace("_", " ").replace("-", " ").lower().split())
    aliases = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "adjusted close": "Adj Close",
        "volume": "Volume",
    }
    return aliases.get(normalized)


def _flatten_ohlcv_columns(columns: pd.Index) -> list[object]:
    """Choose the MultiIndex level that contains OHLCV field names."""

    if not isinstance(columns, pd.MultiIndex):
        return list(columns)

    expected = set(CSV_COLUMNS)
    level_scores = [
        sum(
            _canonical_ohlcv_label(value) in expected
            for value in columns.get_level_values(level)
        )
        for level in range(columns.nlevels)
    ]
    best_level = max(range(columns.nlevels), key=lambda level: level_scores[level])
    if level_scores[best_level] == 0:
        return list(columns)
    return list(columns.get_level_values(best_level))


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily OHLCV columns and index."""

    frame = data.copy()
    frame.columns = _flatten_ohlcv_columns(frame.columns)
    canonical_labels = [_canonical_ohlcv_label(column) for column in frame.columns]

    date_positions = [
        position for position, column in enumerate(canonical_labels) if column == "Date"
    ]
    if date_positions:
        frame.index = pd.to_datetime(frame.iloc[:, date_positions[-1]])
    else:
        frame.index = pd.to_datetime(frame.index)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"

    column_positions: dict[str, int] = {}
    for position, column in enumerate(canonical_labels):
        if column in REQUIRED_OHLCV_COLUMNS:
            # Duplicate OHLCV fields can come from cached CSV headers or yfinance
            # MultiIndex frames; keep the last normalized occurrence deterministically.
            column_positions[column] = position

    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in column_positions]
    if missing:
        raise ValueError(f"OHLCV data missing required columns: {missing}")

    output = pd.DataFrame(index=frame.index)
    for column in REQUIRED_OHLCV_COLUMNS:
        output[column] = pd.to_numeric(
            frame.iloc[:, column_positions[column]],
            errors="coerce",
        )

    if not output.columns.is_unique:
        raise ValueError("OHLCV data has duplicate columns after normalization.")

    output = output.sort_index()
    output = output[~output.index.duplicated(keep="last")]
    return output.dropna(subset=["Adj Close"])


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
        if cached is not None:
            cached = normalize_ohlcv(cached)
            if cache_satisfies(cached, start, end):
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
