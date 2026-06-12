from __future__ import annotations

import pandas as pd

from src.data.cache import cache_path
from src.data.fetch import load_daily_data
from src.features.technical import build_technical_features


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

    features = build_technical_features(data, benchmark_frames={"SPY": benchmark})

    assert "rs_spy_60d" in features
