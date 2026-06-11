from __future__ import annotations

import pandas as pd

from src.features.regime import DOWNTREND_HIGH_RISK, UPTREND_LOW_VOL
from src.portfolio.allocation import AllocationConfig, allocate_from_scores


def test_allocation_rule_enforces_max_position_and_cash_floor() -> None:
    scores = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "ML Score": [95.0, 75.0, 10.0],
            "ML Drawdown-Risk Probability": [0.10, 0.30, 0.20],
            "Rule-Based Regime": [UPTREND_LOW_VOL, UPTREND_LOW_VOL, UPTREND_LOW_VOL],
        }
    )
    config = AllocationConfig(max_position_size=0.20, cash_floor=0.50, max_gross_exposure=0.80)

    allocation = allocate_from_scores(scores, config)

    assert allocation["target_weight"].max() <= 0.20
    assert allocation["target_weight"].sum() <= 0.50
    assert allocation.loc[allocation["Ticker"] == "CCC", "target_weight"].iloc[0] == 0.0


def test_allocation_exits_high_drawdown_or_downtrend() -> None:
    scores = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "ML Score": [90.0, 90.0],
            "ML Drawdown-Risk Probability": [0.80, 0.10],
            "Rule-Based Regime": [UPTREND_LOW_VOL, DOWNTREND_HIGH_RISK],
        }
    )

    allocation = allocate_from_scores(scores, AllocationConfig(drawdown_risk_threshold=0.60))

    assert allocation["target_weight"].sum() == 0.0
    assert set(allocation["suggested_action"]) == {"Watch"}

