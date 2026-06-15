from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.fourier import rolling_fourier_features
from src.features.technical import build_technical_features, rsi_momentum
from src.features.wavelet import rolling_wavelet_features


def synthetic_ohlcv(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    price = pd.Series(np.linspace(100, 150, rows), index=index)
    return pd.DataFrame(
        {
            "Open": price,
            "High": price + 1,
            "Low": price - 1,
            "Close": price,
            "Adj Close": price,
            "Volume": np.linspace(1_000_000, 2_000_000, rows),
        },
        index=index,
    )


def test_technical_feature_shape_and_missing_data() -> None:
    data = synthetic_ohlcv()
    data.loc[data.index[10], "Adj Close"] = np.nan

    features = build_technical_features(data.dropna(subset=["Adj Close"]))

    assert len(features) == len(data) - 1
    assert "return_60d" in features
    assert "volatility_20d" in features
    assert "dist_ma_200d" in features
    assert features["return_5d"].notna().sum() > 0


def test_rsi_momentum_obvious_uptrend() -> None:
    price = pd.Series(np.arange(1, 40), index=pd.date_range("2021-01-01", periods=39))
    rsi = rsi_momentum(price, window=14)
    assert rsi.dropna().iloc[-1] == 100.0


def test_fourier_feature_columns_and_window() -> None:
    data = synthetic_ohlcv(120)
    features = rolling_fourier_features(data["Adj Close"], window=40, n_components=2)

    assert features.shape == (120, 11)
    assert features.iloc[:40].drop(columns=[]).isna().all().all()
    assert features["fourier_energy_concentration"].notna().sum() > 0
    assert "fourier_cycle_clarity" in features
    assert "fourier_cycle_strength" in features
    assert "fourier_noise_diffusion" in features
    derived = features[
        ["fourier_cycle_clarity", "fourier_cycle_strength", "fourier_noise_diffusion"]
    ].dropna()
    assert not derived.empty
    assert np.isfinite(derived.to_numpy()).all()


def test_wavelet_feature_columns() -> None:
    data = synthetic_ohlcv(140)
    features = rolling_wavelet_features(data["Adj Close"], window=64, level=3)

    assert "wavelet_available" in features
    assert "wavelet_trend_return" in features
    assert "wavelet_clean_trend" in features
    assert "wavelet_trend_quality" in features
    assert "wavelet_noise_pressure" in features
    assert "wavelet_medium_long_energy_share" in features
    assert features.shape[0] == 140
    assert features.iloc[:63].drop(columns=["wavelet_available"]).isna().all().all()
    derived = features[
        [
            "wavelet_clean_trend",
            "wavelet_trend_quality",
            "wavelet_noise_pressure",
            "wavelet_medium_long_energy_share",
        ]
    ].dropna()
    assert not derived.empty
    assert np.isfinite(derived.to_numpy()).all()
