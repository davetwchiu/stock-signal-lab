from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.regime import DISTRIBUTION, UPTREND_LOW_VOL
from src.ml.labels import (
    forward_drawdown,
    forward_regime_deterioration,
    make_forward_labels,
    make_risk_adjusted_relative_forward_target,
)


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


def test_risk_adjusted_relative_target_rewards_benchmark_outperformance() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    price = pd.Series([100, 102, 110, 112, 113, 114, 115, 116], index=index, dtype=float)
    benchmark = pd.Series([100, 101, 104, 105, 106, 107, 108, 109], index=index, dtype=float)

    target = make_risk_adjusted_relative_forward_target(price, benchmark, horizon=2)

    expected_excess = (price.iloc[2] / price.iloc[0] - 1.0) - (benchmark.iloc[2] / benchmark.iloc[0] - 1.0)
    assert target.iloc[0] == pytest.approx(expected_excess)
    assert target.iloc[0] > 0


def test_risk_adjusted_relative_target_penalizes_forward_drawdown() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    smooth_price = pd.Series([100, 104, 108, 109, 110, 111, 112, 113], index=index, dtype=float)
    volatile_price = pd.Series([100, 90, 108, 109, 110, 111, 112, 113], index=index, dtype=float)
    benchmark = pd.Series(100.0, index=index)

    smooth_target = make_risk_adjusted_relative_forward_target(smooth_price, benchmark, horizon=2)
    volatile_target = make_risk_adjusted_relative_forward_target(volatile_price, benchmark, horizon=2)

    assert volatile_target.iloc[0] < smooth_target.iloc[0]
    assert volatile_target.iloc[0] == pytest.approx(0.08 - 0.5 * 0.10)


def test_risk_adjusted_relative_target_handles_missing_and_short_data() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    price = pd.Series([100, np.nan, 102, 103], index=index, dtype=float)
    benchmark = pd.Series([100, 101, np.nan, 103], index=index, dtype=float)

    target = make_risk_adjusted_relative_forward_target(price, benchmark, horizon=3)

    assert np.isfinite(target.dropna()).all()
    assert target.tail(3).isna().all()


def test_risk_adjusted_relative_target_does_not_mutate_inputs() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    price = pd.Series([100, 95, 105, 106, 107, 108], index=index, dtype=float)
    benchmark = pd.Series(100.0, index=index)
    original_price = price.copy(deep=True)
    original_benchmark = benchmark.copy(deep=True)

    make_risk_adjusted_relative_forward_target(price, benchmark, horizon=2)

    pd.testing.assert_series_equal(price, original_price)
    pd.testing.assert_series_equal(benchmark, original_benchmark)


def test_make_forward_labels_includes_risk_adjusted_v2_columns() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    price = pd.Series([100, 90, 108, 109, 110, 111, 112, 113], index=index, dtype=float)
    benchmark = pd.Series(100.0, index=index)
    features = pd.DataFrame({"Adj Close": price}, index=index)

    labels = make_forward_labels(features, benchmark, horizon=2, outperformance_threshold=0.02)

    assert labels.loc[index[0], "forward_2d_risk_adjusted_excess_return"] == pytest.approx(0.03)
    assert labels.loc[index[0], "label_risk_adjusted_outperform_2d"] == 1.0
    assert labels["label_risk_adjusted_outperform_2d"].tail(2).isna().all()
