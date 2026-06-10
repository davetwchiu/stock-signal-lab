"""No-lookahead portfolio simulator for Stage 3 research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.portfolio.allocation import AllocationConfig, allocate_from_scores
from src.portfolio.metrics import contribution_by_ticker, summarize_portfolio
from src.portfolio.risk import RiskControlConfig, apply_portfolio_risk_controls


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Portfolio simulation outputs."""

    curve: pd.DataFrame
    weights: pd.DataFrame
    target_weights: pd.DataFrame
    summary: pd.DataFrame
    contribution: pd.DataFrame
    target_log: pd.DataFrame


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "weekly") -> pd.DatetimeIndex:
    """Return dates on which target weights may change."""

    if len(index) == 0:
        return pd.DatetimeIndex([])
    freq = frequency.lower()
    series = pd.Series(index=index, data=index)
    if freq == "monthly":
        return pd.DatetimeIndex(series.groupby([index.year, index.month]).tail(1).values)
    if freq == "weekly":
        calendar = index.to_series()
        return pd.DatetimeIndex(series.groupby([calendar.dt.isocalendar().year, calendar.dt.isocalendar().week]).tail(1).values)
    raise ValueError("frequency must be 'weekly' or 'monthly'")


def _price_frame(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prices = {
        ticker: frame["Adj Close"]
        for ticker, frame in price_frames.items()
        if not frame.empty and "Adj Close" in frame
    }
    return pd.DataFrame(prices).sort_index().ffill()


def simulate_portfolio(
    price_frames: dict[str, pd.DataFrame],
    score_panel: pd.DataFrame,
    allocation_config: AllocationConfig | None = None,
    risk_config: RiskControlConfig | None = None,
    benchmark_features: pd.DataFrame | None = None,
    benchmark_price: pd.Series | None = None,
    rebalance_frequency: str = "weekly",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    starting_capital: float = 100_000.0,
) -> PortfolioBacktestResult:
    """Simulate a long-only portfolio with targets lagged one day."""

    prices = _price_frame(price_frames)
    if prices.empty or score_panel.empty:
        empty = pd.DataFrame()
        return PortfolioBacktestResult(empty, empty, empty, empty, empty, empty)

    cfg = allocation_config or AllocationConfig()
    risk_cfg = risk_config or RiskControlConfig(
        cash_floor=cfg.cash_floor,
        max_gross_exposure=cfg.max_gross_exposure,
    )
    returns = prices.pct_change().fillna(0.0)
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    logs: list[pd.DataFrame] = []
    current_target = pd.Series(0.0, index=prices.columns)
    rebalances = set(rebalance_dates(prices.index, rebalance_frequency))
    running_equity = starting_capital
    peak_equity = starting_capital

    score_panel = score_panel.copy()
    score_panel["Date"] = pd.to_datetime(score_panel["Date"])

    for date in prices.index:
        if date in rebalances:
            latest_scores = score_panel[score_panel["Date"] <= date].sort_values("Date").groupby("Ticker").tail(1)
            latest_scores = latest_scores[latest_scores["Ticker"].isin(prices.columns)]
            allocation = allocate_from_scores(latest_scores, cfg, current_weights=current_target)
            next_target = pd.to_numeric(
                allocation.set_index("Ticker")["target_weight"].reindex(prices.columns),
                errors="coerce",
            ).fillna(0.0)
            benchmark_row = None
            if benchmark_features is not None and not benchmark_features.empty:
                eligible = benchmark_features.loc[benchmark_features.index <= date]
                if not eligible.empty:
                    benchmark_row = eligible.iloc[-1]
            portfolio_drawdown = running_equity / peak_equity - 1.0
            next_target, flags = apply_portfolio_risk_controls(
                next_target,
                risk_cfg,
                benchmark_row=benchmark_row,
                portfolio_drawdown=portfolio_drawdown,
            )
            current_target = next_target
            log = allocation.copy()
            log["Date"] = date
            log["risk_control_flags"] = ", ".join(flags)
            logs.append(log)
        target_weights.loc[date] = current_target
        effective_weights = target_weights.shift(1).fillna(0.0)
        day_return = float((effective_weights.loc[date] * returns.loc[date]).sum())
        running_equity *= 1.0 + day_return
        peak_equity = max(peak_equity, running_equity)

    weights = target_weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    gross_return = (weights * returns).sum(axis=1)
    trading_cost = turnover * cost_rate
    portfolio_return = gross_return - trading_cost

    curve = pd.DataFrame(index=prices.index)
    curve["portfolio_return"] = portfolio_return
    curve["portfolio_equity"] = starting_capital * (1.0 + portfolio_return).cumprod()
    curve["turnover"] = turnover
    curve["trading_cost"] = trading_cost
    curve["gross_exposure"] = weights.sum(axis=1)
    curve["cash"] = 1.0 - curve["gross_exposure"]
    curve["holdings"] = (weights > 0).sum(axis=1)
    if benchmark_price is not None:
        benchmark = benchmark_price.reindex(curve.index).ffill()
        curve["benchmark_equity"] = starting_capital * (1.0 + benchmark.pct_change().fillna(0.0)).cumprod()

    contribution = contribution_by_ticker(weights, returns)
    summary = summarize_portfolio(curve, weights, contribution)
    target_log = pd.concat([log.dropna(axis=1, how="all") for log in logs], ignore_index=True) if logs else pd.DataFrame()
    return PortfolioBacktestResult(curve, weights, target_weights, summary, contribution, target_log)
