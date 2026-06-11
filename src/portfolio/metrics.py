"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio, win_rate


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    """CAGR divided by absolute max drawdown."""

    cagr = annualized_return(returns)
    drawdown = abs(max_drawdown(equity))
    if drawdown == 0 or pd.isna(drawdown):
        return np.nan
    return float(cagr / drawdown)


def contribution_by_ticker(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate return and drawdown contribution by ticker."""

    aligned_returns = returns.reindex(weights.index).fillna(0.0)
    contribution = weights.shift(1).fillna(0.0) * aligned_returns
    total = contribution.sum()
    downside = contribution.where(contribution < 0, 0.0).sum()
    return pd.DataFrame(
        {
            "Ticker": total.index,
            "Return Contribution": total.values,
            "Downside Contribution": downside.values,
        }
    ).sort_values("Return Contribution", ascending=False)


def summarize_portfolio(
    curve: pd.DataFrame,
    weights: pd.DataFrame,
    contribution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize portfolio performance and operating metrics."""

    returns = curve["portfolio_return"].dropna()
    equity = curve["portfolio_equity"].dropna()
    row = {
        "CAGR": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(equity) if not equity.empty else np.nan,
        "Calmar": calmar_ratio(returns, equity) if not equity.empty else np.nan,
        "Turnover": float(curve.get("turnover", pd.Series(dtype=float)).sum()),
        "Average Cash": float(curve.get("cash", pd.Series(dtype=float)).mean()),
        "Average Exposure": float(curve.get("gross_exposure", pd.Series(dtype=float)).mean()),
        "Average Holdings": float((weights > 0).sum(axis=1).mean()) if not weights.empty else np.nan,
        "Hit Rate": win_rate(returns),
    }
    if contribution is not None and not contribution.empty:
        row["Top Contributor"] = contribution.iloc[0]["Ticker"]
    return pd.DataFrame([row], index=["Portfolio"])

