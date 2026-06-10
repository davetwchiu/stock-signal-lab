"""Performance metric calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown for an equity curve."""

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from periodic returns."""

    clean = returns.dropna()
    if clean.empty:
        return np.nan
    total = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""

    return float(returns.dropna().std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio with zero risk-free rate."""

    clean = returns.dropna()
    vol = clean.std()
    if clean.empty or vol == 0 or pd.isna(vol):
        return np.nan
    return float(clean.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio with zero risk-free rate."""

    clean = returns.dropna()
    downside = clean[clean < 0].std()
    if clean.empty or downside == 0 or pd.isna(downside):
        return np.nan
    return float(clean.mean() / downside * np.sqrt(periods_per_year))


def win_rate(returns: pd.Series) -> float:
    """Share of non-zero return days that are positive."""

    active = returns.dropna()
    active = active[active != 0]
    if active.empty:
        return np.nan
    return float((active > 0).mean())


def summarize_performance(
    returns: pd.Series,
    equity: pd.Series,
    positions: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    """Return common performance and risk metrics."""

    clean_returns = returns.dropna()
    summary = {
        "CAGR": annualized_return(clean_returns),
        "Annualized Volatility": annualized_volatility(clean_returns),
        "Sharpe": sharpe_ratio(clean_returns),
        "Sortino": sortino_ratio(clean_returns),
        "Max Drawdown": max_drawdown(equity.dropna()) if not equity.dropna().empty else np.nan,
        "Win Rate": win_rate(clean_returns),
    }
    if turnover is not None:
        summary["Turnover"] = float(turnover.dropna().sum())
    if positions is not None:
        summary["Exposure"] = float((positions.dropna() > 0).mean())
    return summary

