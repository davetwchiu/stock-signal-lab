"""Decision support for the validated S&P sector broad-pullback state."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SECTOR_PULLBACK_TICKERS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")


@dataclass(frozen=True)
class BroadPullbackPlaybook:
    status: str
    action: str
    as_of: str
    negative_sector_count: int
    available_sector_count: int
    benchmark_trend: str
    evidence: str


def build_broad_pullback_playbook(frames: dict[str, pd.DataFrame]) -> BroadPullbackPlaybook:
    """Apply the exact validated rule without changing portfolio actions."""

    spy = _price(frames.get("SPY"))
    if len(spy) < 6:
        return _unavailable("SPY needs at least six aligned trading days.")

    as_of = spy.index[-1]
    signal_dates = spy.tail(6).index
    sector_returns: list[float] = []
    for ticker in SECTOR_PULLBACK_TICKERS:
        prices = _price(frames.get(ticker)).reindex(signal_dates)
        if prices.isna().any() or (prices <= 0).any():
            continue
        sector_returns.append(float(prices.iloc[-1] / prices.iloc[0] - 1.0))

    available_count = len(sector_returns)
    negative_count = sum(value < 0 for value in sector_returns)
    as_of_label = as_of.strftime("%Y-%m-%d")
    if available_count < len(SECTOR_PULLBACK_TICKERS):
        return _unavailable(
            f"Only {available_count}/9 sector ETFs have aligned 5-day prices; the exact rule cannot be tested.",
            as_of=as_of_label,
            negative_count=negative_count,
            available_count=available_count,
        )

    spy_history = spy.tail(200)
    if len(spy_history) < 200:
        return _unavailable(
            "SPY needs 200 trading days of history before the validated regime filter can be tested.",
            as_of=as_of_label,
            negative_count=negative_count,
            available_count=available_count,
        )

    above_200dma = bool(spy_history.iloc[-1] >= spy_history.mean())
    benchmark_trend = "above 200-day average" if above_200dma else "below 200-day average"
    if negative_count == 9 and above_200dma:
        return BroadPullbackPlaybook(
            status="Confirmed",
            action="Hold / review position size",
            as_of=as_of_label,
            negative_sector_count=negative_count,
            available_sector_count=available_count,
            benchmark_trend=benchmark_trend,
            evidence=(
                "All nine sectors are down over five trading days while SPY remains above its 200-day average. "
                "Across 85 non-overlapping normal-market historical events, keeping SPY exposure beat selling "
                "SPY to cash for 20 trading days in 72.9% of cases; median estimated sell-and-rebuy regret was "
                "1.75% after 10 bp round-trip cost. Do not treat this broad-market state as an exit signal; "
                "this is not an automatic buy signal or a promise of a quick recovery."
            ),
        )

    condition = (
        f"{negative_count}/9 sectors are down over five trading days"
        if negative_count < 9
        else "All nine sectors are down, but SPY is below its 200-day average"
    )
    return BroadPullbackPlaybook(
        status="Not confirmed",
        action="No playbook signal",
        as_of=as_of_label,
        negative_sector_count=negative_count,
        available_sector_count=available_count,
        benchmark_trend=benchmark_trend,
        evidence=f"{condition}; the validated broad-pullback statistics do not apply.",
    )


def _price(frame: pd.DataFrame | None) -> pd.Series:
    if frame is None or frame.empty or "Adj Close" not in frame:
        return pd.Series(dtype=float)
    price = pd.to_numeric(frame["Adj Close"], errors="coerce").dropna()
    price.index = pd.to_datetime(price.index)
    return price[~price.index.duplicated(keep="last")].sort_index()


def _unavailable(
    evidence: str,
    *,
    as_of: str = "Unavailable",
    negative_count: int = 0,
    available_count: int = 0,
) -> BroadPullbackPlaybook:
    return BroadPullbackPlaybook(
        status="Unavailable",
        action="No playbook signal",
        as_of=as_of,
        negative_sector_count=negative_count,
        available_sector_count=available_count,
        benchmark_trend="unavailable",
        evidence=evidence,
    )
