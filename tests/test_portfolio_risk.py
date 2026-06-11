from __future__ import annotations

import pandas as pd

from src.portfolio.risk import RiskControlConfig, apply_portfolio_risk_controls, benchmark_is_downtrend


def test_benchmark_downtrend_detection_and_derisking() -> None:
    weights = pd.Series({"AAA": 0.20, "BBB": 0.20})
    benchmark_row = pd.Series({"regime": "Downtrend / high risk", "dist_ma_200d": -0.05, "return_60d": -0.04})

    adjusted, flags = apply_portfolio_risk_controls(weights, RiskControlConfig(), benchmark_row=benchmark_row)

    assert benchmark_is_downtrend(benchmark_row)
    assert adjusted.sum() < weights.sum()
    assert "benchmark downtrend de-risking" in flags


def test_portfolio_drawdown_risk_control_triggers() -> None:
    weights = pd.Series({"AAA": 0.20, "BBB": 0.20})
    config = RiskControlConfig(portfolio_drawdown_limit=-0.10, portfolio_drawdown_multiplier=0.25)

    adjusted, flags = apply_portfolio_risk_controls(weights, config, portfolio_drawdown=-0.20)

    assert adjusted.sum() == 0.10
    assert "portfolio drawdown de-risking" in flags

