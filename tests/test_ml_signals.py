from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.ml.signals import ml_probability_signal


def test_ml_signal_rule_and_backtest_lag() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    out_prob = pd.Series([0.7, 0.7, 0.4, 0.8], index=index)
    risk_prob = pd.Series([0.2, 0.6, 0.2, 0.3], index=index)
    price = pd.Series([100.0, 110.0, 121.0, 121.0], index=index)

    signal = ml_probability_signal(out_prob, risk_prob, outperform_threshold=0.6, drawdown_risk_threshold=0.4)
    result = run_backtest(price, signal, transaction_cost_bps=0.0, slippage_bps=0.0)

    assert signal.tolist() == [1.0, 0.25, 0.0, 1.0]
    assert result.curve["position"].tolist() == [0.0, 1.0, 0.25, 0.0]
    assert np.isclose(result.curve["strategy_return"].iloc[1], 0.10)
