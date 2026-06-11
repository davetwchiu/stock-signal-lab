"""Simple portfolio risk controls."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskControlConfig:
    """Portfolio-level risk controls used by the simulator."""

    cash_floor: float = 0.10
    max_gross_exposure: float = 0.90
    benchmark_downtrend_multiplier: float = 0.50
    portfolio_drawdown_limit: float = -0.15
    portfolio_drawdown_multiplier: float = 0.50
    min_holding_period: int = 0
    no_rebuy_cooldown: int = 0


def benchmark_is_downtrend(row: pd.Series | None) -> bool:
    """Return whether benchmark features indicate a downtrend."""

    if row is None or row.empty:
        return False
    regime = str(row.get("regime", ""))
    below_200 = row.get("dist_ma_200d", 0.0)
    trend_60 = row.get("return_60d", 0.0)
    return "Downtrend" in regime or (pd.notna(below_200) and below_200 < 0) or (pd.notna(trend_60) and trend_60 < 0)


def apply_portfolio_risk_controls(
    weights: pd.Series,
    config: RiskControlConfig | None = None,
    benchmark_row: pd.Series | None = None,
    portfolio_drawdown: float = 0.0,
) -> tuple[pd.Series, list[str]]:
    """Apply portfolio-level de-risking and exposure caps."""

    cfg = config or RiskControlConfig()
    adjusted = weights.clip(lower=0.0).copy()
    flags: list[str] = []

    if benchmark_is_downtrend(benchmark_row):
        adjusted *= cfg.benchmark_downtrend_multiplier
        flags.append("benchmark downtrend de-risking")
    if portfolio_drawdown <= cfg.portfolio_drawdown_limit:
        adjusted *= cfg.portfolio_drawdown_multiplier
        flags.append("portfolio drawdown de-risking")

    exposure_cap = min(cfg.max_gross_exposure, 1.0 - cfg.cash_floor)
    gross = float(adjusted.sum())
    if gross > exposure_cap and gross > 0:
        adjusted *= exposure_cap / gross
        flags.append("gross exposure cap")

    return adjusted, flags

