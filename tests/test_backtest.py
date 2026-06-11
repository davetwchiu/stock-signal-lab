from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.metrics import max_drawdown, summarize_performance
from src.backtest.signals import lag_positions


def test_signal_lagging() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    raw = pd.Series([1.0, 0.0, 1.0, 1.0], index=index)

    lagged = lag_positions(raw)

    assert lagged.iloc[0] == 0.0
    assert lagged.iloc[1] == 1.0
    assert lagged.iloc[2] == 0.0


def test_backtest_captures_next_close_return_after_lag() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    price = pd.Series([100.0, 110.0, 110.0], index=index)
    signal = pd.Series([1.0, 0.0, 0.0], index=index)

    result = run_backtest(price, signal, transaction_cost_bps=0.0, slippage_bps=0.0)

    assert result.curve["position"].tolist() == [0.0, 1.0, 0.0]
    assert result.curve["strategy_equity"].iloc[-1] == 1.1


def test_backtest_accounting_includes_turnover_cost() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    price = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    signal = pd.Series([1.0, 1.0, 0.0, 0.0], index=index)

    result = run_backtest(price, signal, transaction_cost_bps=10.0, slippage_bps=0.0)

    assert result.curve["turnover"].sum() == 2.0
    assert result.curve["strategy_equity"].iloc[-1] < 1.0


def test_metrics_calculation() -> None:
    returns = pd.Series([0.1, -0.05, 0.02, 0.0])
    equity = (1 + returns).cumprod()
    summary = summarize_performance(returns, equity)

    assert np.isclose(max_drawdown(equity), -0.05)
    assert "Sharpe" in summary
    assert summary["Win Rate"] == 2 / 3

