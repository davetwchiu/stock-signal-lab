from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.regime import DISTRIBUTION, UPTREND_LOW_VOL
from src.ml.labels import forward_drawdown, forward_regime_deterioration, make_forward_labels


def test_forward_outperformance_label_alignment() -> None:
    index = pd.date_range("2024-01-01", periods=25, freq="B")
    price = pd.Series(np.arange(100, 125, dtype=float), index=index)
    benchmark = pd.Series(100.0, index=index)
    features = pd.DataFrame({"Adj Close": price}, index=index)

    labels = make_forward_labels(features, benchmark, horizon=5, outperformance_threshold=0.02)

    expected_return = price.iloc[5] / price.iloc[0] - 1.0
    assert labels.loc[index[0], "forward_5d_return"] == expected_return
    assert labels.loc[index[0], "label_outperform_5d"] == 1.0
    assert labels["label_outperform_5d"].tail(5).isna().all()


def test_forward_drawdown_uses_next_window_only() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    price = pd.Series([100, 110, 90, 120, 130, 140, 150, 160], index=index, dtype=float)

    drawdown = forward_drawdown(price, horizon=3)

    assert np.isclose(drawdown.iloc[0], -0.10)
    assert drawdown.tail(3).isna().all()


def test_forward_regime_deterioration_label() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    regimes = pd.Series(
        [UPTREND_LOW_VOL, UPTREND_LOW_VOL, DISTRIBUTION, UPTREND_LOW_VOL, UPTREND_LOW_VOL, UPTREND_LOW_VOL],
        index=index,
    )

    labels = forward_regime_deterioration(regimes, horizon=2)

    assert labels.iloc[0] == 1.0
    assert labels.iloc[2] == 0.0
    assert labels.tail(2).isna().all()
