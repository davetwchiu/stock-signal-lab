from __future__ import annotations

import pandas as pd
import pytest

from src.research.backtest import (
    BUY_AND_HOLD,
    SIMPLE_TREND,
    SSL_PRODUCTION,
    SMA_200,
    SMA_50_200,
    SimpleROIBacktestConfig,
    add_comparison_columns,
    buy_and_hold_signal,
    drawdown_details,
    moving_average_cross_signal,
    moving_average_signal,
    run_single_strategy_backtest,
    simple_trend_drawdown_risk_signal,
    simple_trend_reduced_defensiveness_signal,
    simple_trend_signal,
)


def price_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=index, dtype=float)


def test_buy_and_hold_math() -> None:
    price = price_series([100.0, 110.0, 121.0])

    result = run_single_strategy_backtest("AAA", price, buy_and_hold_signal(price), BUY_AND_HOLD)

    assert result.summary["final_value"] == pytest.approx(1210.0)
    assert result.summary["total_return"] == pytest.approx(0.21)


def test_200dma_signal_behaviour() -> None:
    price = price_series([100.0] * 200 + [101.0])

    signal = moving_average_signal(price, window=200)

    assert signal.iloc[199] == 0.0
    assert signal.iloc[200] == 1.0


def test_50_200dma_cross_behaviour() -> None:
    price = price_series([100.0] * 150 + [120.0] * 51)

    signal = moving_average_cross_signal(price, short_window=50, long_window=200)

    assert signal.iloc[198] == 0.0
    assert signal.iloc[200] == 1.0


def test_simple_trend_rules_are_long_cash_only() -> None:
    features = pd.DataFrame(
        {
            "dist_ma_50d": [0.01, -0.01, 0.02, -0.01],
            "dist_ma_200d": [0.02, 0.02, 0.03, -0.02],
            "max_drawdown_60d": [-0.05, -0.05, -0.15, -0.05],
            "max_drawdown_120d": [-0.10, -0.10, -0.10, -0.25],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )

    assert simple_trend_signal(features).tolist() == [1.0, 0.0, 1.0, 0.0]
    assert simple_trend_drawdown_risk_signal(features).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert simple_trend_reduced_defensiveness_signal(features).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_no_lookahead_behaviour() -> None:
    price = price_series([100.0, 200.0, 200.0])
    same_day_signal = pd.Series([0.0, 1.0, 0.0], index=price.index)

    result = run_single_strategy_backtest("AAA", price, same_day_signal, SMA_200)

    assert result.curve["position"].tolist() == [0.0, 0.0, 1.0]
    assert result.summary["final_value"] == pytest.approx(1000.0)


def test_max_drawdown_calculation_includes_start_and_end() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    equity = pd.Series([1000.0, 1200.0, 900.0, 1300.0], index=index)

    details = drawdown_details(equity)

    assert details["max_drawdown"] == pytest.approx(-0.25)
    assert details["start"] == index[1]
    assert details["end"] == index[2]


def test_trade_count_and_days_in_market_calculation() -> None:
    price = price_series([100.0, 101.0, 102.0, 103.0])
    raw_signal = pd.Series([1.0, 1.0, 0.0, 1.0], index=price.index)

    result = run_single_strategy_backtest("AAA", price, raw_signal, SMA_50_200)

    assert result.curve["position"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result.summary["number_of_trades"] == 2
    assert result.summary["days_in_market"] == 2
    assert result.summary["percent_days_in_market"] == pytest.approx(0.5)


def test_transaction_cost_defaults_to_zero() -> None:
    assert SimpleROIBacktestConfig().transaction_cost_bps == 0.0


def test_fractional_share_portfolio_value_math() -> None:
    price = price_series([300.0, 330.0])

    result = run_single_strategy_backtest("AAA", price, buy_and_hold_signal(price), BUY_AND_HOLD)

    assert result.curve["fractional_shares"].iloc[1] == pytest.approx(1000.0 / 300.0)
    assert result.summary["final_value"] == pytest.approx(1100.0)


def test_comparison_columns_are_per_strategy() -> None:
    results = pd.DataFrame(
        [
            {"ticker": "AAA", "strategy": BUY_AND_HOLD, "final_value": 1100.0, "total_return": 0.10, "max_drawdown": -0.20},
            {"ticker": "AAA", "strategy": SMA_200, "final_value": 1050.0, "total_return": 0.05, "max_drawdown": -0.05},
            {
                "ticker": "AAA",
                "strategy": SSL_PRODUCTION,
                "final_value": 1020.0,
                "total_return": 0.02,
                "max_drawdown": -0.02,
            },
            {
                "ticker": "AAA",
                "strategy": SIMPLE_TREND,
                "final_value": 1080.0,
                "total_return": 0.08,
                "max_drawdown": -0.04,
            },
        ]
    )

    output = add_comparison_columns(results).set_index("strategy")

    assert output.loc[SIMPLE_TREND, "beats_buy_hold"] is False
    assert output.loc[SIMPLE_TREND, "beats_200dma"] is True
    assert output.loc[SIMPLE_TREND, "improves_return_drawdown_tradeoff_vs_ssl"] is True
