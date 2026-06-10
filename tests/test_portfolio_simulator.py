from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.allocation import AllocationConfig
from src.portfolio.simulator import rebalance_dates, simulate_portfolio


def price_frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    price = pd.Series(values, index=index)
    return pd.DataFrame({"Adj Close": price})


def test_rebalance_timing_weekly() -> None:
    index = pd.date_range("2024-01-01", periods=15, freq="B")
    dates = rebalance_dates(index, "weekly")

    assert len(dates) == 3
    assert dates[0].weekday() == 4


def test_portfolio_simulation_lags_targets_and_handles_costs() -> None:
    prices = {"AAA": price_frame([100, 110, 121, 121, 121, 130]), "BBB": price_frame([100, 100, 100, 100, 100, 100])}
    first_date = next(iter(prices.values())).index[0]
    score_panel = pd.DataFrame(
        {
            "Date": [first_date],
            "Ticker": ["AAA"],
            "ML Score": [95.0],
            "ML Drawdown-Risk Probability": [0.10],
            "Rule-Based Regime": ["Uptrend / low volatility"],
        }
    )

    no_cost = simulate_portfolio(
        prices,
        score_panel,
        allocation_config=AllocationConfig(max_position_size=1.0, cash_floor=0.0, max_gross_exposure=1.0),
        rebalance_frequency="weekly",
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    with_cost = simulate_portfolio(
        prices,
        score_panel,
        allocation_config=AllocationConfig(max_position_size=1.0, cash_floor=0.0, max_gross_exposure=1.0),
        rebalance_frequency="weekly",
        transaction_cost_bps=10.0,
        slippage_bps=0.0,
    )

    assert no_cost.weights.iloc[0].sum() == 0.0
    assert no_cost.weights.iloc[-1]["AAA"] == 1.0
    assert with_cost.curve["portfolio_equity"].iloc[-1] < no_cost.curve["portfolio_equity"].iloc[-1]


def test_future_score_mutation_does_not_change_prior_weights() -> None:
    prices = {"AAA": price_frame([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])}
    index = next(iter(prices.values())).index
    score_panel = pd.DataFrame(
        {
            "Date": [index[4], index[9]],
            "Ticker": ["AAA", "AAA"],
            "ML Score": [95.0, 0.0],
            "ML Drawdown-Risk Probability": [0.10, 0.90],
            "Rule-Based Regime": ["Uptrend / low volatility", "Downtrend / high risk"],
        }
    )
    changed = score_panel.copy()
    changed.loc[changed["Date"] == index[9], "ML Score"] = 100.0

    original = simulate_portfolio(prices, score_panel, rebalance_frequency="weekly")
    mutated = simulate_portfolio(prices, changed, rebalance_frequency="weekly")

    pd.testing.assert_frame_equal(original.weights.loc[: index[8]], mutated.weights.loc[: index[8]])
