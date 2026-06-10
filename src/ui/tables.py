"""Table builders for overview, regimes, and rankings."""

from __future__ import annotations

import pandas as pd


def current_regime_table(feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the latest available regime row for each ticker."""

    rows: list[dict[str, object]] = []
    for ticker, frame in feature_frames.items():
        if frame.empty:
            continue
        latest = frame.dropna(subset=["Adj Close"]).iloc[-1]
        rows.append(
            {
                "Ticker": ticker,
                "Date": latest.name,
                "Close": latest.get("Adj Close"),
                "Regime": latest.get("regime"),
                "Risk Flags": latest.get("risk_flags", ""),
                "60d Return": latest.get("return_60d"),
                "120d Return": latest.get("return_120d"),
                "60d Vol": latest.get("volatility_60d"),
                "60d Drawdown": latest.get("max_drawdown_60d"),
                "RS vs SPY 60d": latest.get("rs_spy_60d"),
                "RS vs QQQ 60d": latest.get("rs_qqq_60d"),
                "Rationale": latest.get("regime_rationale"),
            }
        )
    return pd.DataFrame(rows)


def overview_table(feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a compact market overview table."""

    rows: list[dict[str, object]] = []
    for ticker, frame in feature_frames.items():
        if frame.empty:
            continue
        latest = frame.iloc[-1]
        rows.append(
            {
                "Ticker": ticker,
                "Close": latest.get("Adj Close"),
                "Daily Return": latest.get("daily_return"),
                "20d Return": latest.get("return_20d"),
                "60d Return": latest.get("return_60d"),
                "60d Vol": latest.get("volatility_60d"),
                "Volume Z 20d": latest.get("volume_z_20d"),
            }
        )
    return pd.DataFrame(rows)


def relative_strength_ranking(regime_table: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """Rank tickers by 60d relative strength to the selected benchmark."""

    column = f"RS vs {benchmark.upper()} 60d"
    if column not in regime_table:
        return pd.DataFrame()
    return regime_table.sort_values(column, ascending=False, na_position="last")[
        ["Ticker", column, "Regime", "Risk Flags"]
    ].reset_index(drop=True)

