"""No-lookahead close-to-close backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.metrics import summarize_performance
from src.backtest.signals import lag_positions


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    curve: pd.DataFrame
    summary: pd.DataFrame


def run_backtest(
    price: pd.Series,
    target_positions: pd.Series,
    benchmark_price: pd.Series | None = None,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Run a long-only daily close-to-close backtest.

    The raw signal at date `t` is shifted by one row before it is applied to
    returns, which prevents same-day information from earning same-day returns.
    """

    aligned_price = price.dropna().astype(float)
    raw_positions = target_positions.reindex(aligned_price.index).fillna(0.0).astype(float)
    positions = lag_positions(raw_positions)

    asset_returns = aligned_price.pct_change().fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    trading_cost = turnover * cost_rate
    strategy_returns = positions * asset_returns - trading_cost
    buy_hold_returns = asset_returns

    curve = pd.DataFrame(index=aligned_price.index)
    curve["price"] = aligned_price
    curve["asset_return"] = asset_returns
    curve["raw_position"] = raw_positions
    curve["position"] = positions
    curve["turnover"] = turnover
    curve["trading_cost"] = trading_cost
    curve["strategy_return"] = strategy_returns
    curve["buy_hold_return"] = buy_hold_returns
    curve["strategy_equity"] = initial_capital * (1.0 + strategy_returns).cumprod()
    curve["buy_hold_equity"] = initial_capital * (1.0 + buy_hold_returns).cumprod()

    summaries = {
        "Strategy": summarize_performance(strategy_returns, curve["strategy_equity"], positions, turnover),
        "Buy & Hold": summarize_performance(buy_hold_returns, curve["buy_hold_equity"]),
    }

    if benchmark_price is not None:
        benchmark = benchmark_price.reindex(curve.index).ffill()
        benchmark_returns = benchmark.pct_change().fillna(0.0)
        curve["benchmark_return"] = benchmark_returns
        curve["benchmark_equity"] = initial_capital * (1.0 + benchmark_returns).cumprod()
        summaries["Benchmark"] = summarize_performance(benchmark_returns, curve["benchmark_equity"])

    summary = pd.DataFrame(summaries).T
    return BacktestResult(curve=curve, summary=summary)

