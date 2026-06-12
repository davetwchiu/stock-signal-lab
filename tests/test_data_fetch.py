from __future__ import annotations

import pandas as pd

from src.data.cache import cache_path
from src.data.fetch import load_daily_data, normalize_ohlcv
from src.features.technical import build_technical_features


def test_normalize_ohlcv_collapses_duplicate_adj_close_columns() -> None:
    raw = pd.DataFrame(
        [
            ["2020-01-01", 100.0, 101.0, 99.0, 100.0, 10.0, 100.0, 1000],
            ["2020-01-02", 101.0, 102.0, 100.0, 101.0, 11.0, 101.0, 1100],
        ],
        columns=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Adj Close",
            "Volume",
        ],
    )

    normalized = normalize_ohlcv(raw)

    assert list(normalized.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert normalized.columns.is_unique
    assert isinstance(normalized["Adj Close"], pd.Series)
    assert normalized.loc[pd.Timestamp("2020-01-01"), "Adj Close"] == 100.0


def test_load_daily_data_collapses_malformed_cached_adj_close_header(tmp_path) -> None:
    cache_file = cache_path("AAA", tmp_path)
    cache_file.write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Adj Close,Adj Close,Volume",
                "2020-01-01,100,101,99,100,10,100,1000",
                "2020-01-02,101,102,100,101,11,101,1100",
            ]
        ),
        encoding="utf-8",
    )

    data = load_daily_data(
        "AAA",
        start="2020-01-01",
        end="2020-01-02",
        cache_dir=tmp_path,
    )

    assert data.columns.is_unique
    assert isinstance(data["Adj Close"], pd.Series)
    assert data.loc[pd.Timestamp("2020-01-02"), "Adj Close"] == 101


def test_cached_data_duplicate_dates_are_normalized_before_features(tmp_path) -> None:
    cached = pd.DataFrame(
        {
            "Date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
            ],
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Adj Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }
    )
    tmp_path.mkdir(exist_ok=True)
    cached.to_csv(cache_path("AAA", tmp_path), index=False)
    cached.to_csv(cache_path("SPY", tmp_path), index=False)

    data = load_daily_data(
        "AAA",
        start="2020-01-01",
        end="2020-01-06",
        cache_dir=tmp_path,
    )
    benchmark = load_daily_data(
        "SPY",
        start="2020-01-01",
        end="2020-01-06",
        cache_dir=tmp_path,
    )

    assert data.index.is_unique
    assert benchmark.index.is_unique
    assert data.loc[pd.Timestamp("2020-01-02"), "Adj Close"] == 102.0
    assert len(data) == 4

    features = build_technical_features(data, benchmark_frames={"SPY": benchmark})

    assert "rs_spy_60d" in features
