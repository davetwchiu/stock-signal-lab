"""Research-only portfolio overlap and crowding diagnostics."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import ceil

import pandas as pd


ROLLING_WINDOW_DAYS = 60
MIN_CORRELATION_SAMPLES = 20
HIGH_CORRELATION = 0.80
MODERATE_CORRELATION = 0.50
HIGH_AVERAGE_CORRELATION = 0.65
MODERATE_AVERAGE_CORRELATION = 0.45

# ponytail: static proxy map; add real ETF holdings lookthrough only when holdings data exists.
FACTOR_PROXY_MAP: dict[str, tuple[str, str, str]] = {
    "AAPL": ("mega_cap_tech", "mapped", "Apple is a mega-cap technology holding."),
    "GOOG": ("mega_cap_tech", "mapped", "Alphabet is a mega-cap technology holding."),
    "MSFT": ("mega_cap_tech", "mapped", "Microsoft is a mega-cap technology holding."),
    "NVDA": ("semiconductor", "mapped", "NVIDIA is a semiconductor and AI-infrastructure holding."),
    "AVGO": ("semiconductor", "mapped", "Broadcom is a semiconductor and AI-infrastructure holding."),
    "TSM": ("semiconductor", "mapped", "TSMC is a semiconductor foundry holding."),
    "AMAT": ("semiconductor", "mapped", "Applied Materials is a semiconductor equipment holding."),
    "MRVL": ("semiconductor", "mapped", "Marvell is a semiconductor holding."),
    "ANET": ("AI_infrastructure", "mapped", "Arista is an AI-infrastructure networking holding."),
    "PLTR": ("software_AI", "mapped", "Palantir is a software AI holding."),
    "TSLA": ("electric_vehicle", "mapped", "Tesla is an electric-vehicle and technology holding."),
    "KTOS": ("defense", "mapped", "Kratos is a defense technology holding."),
    "ONDS": ("defense", "mapped", "Ondas is mapped as a defense/drone technology proxy."),
    "RDW": ("defense", "mapped", "Redwire is mapped as an aerospace and defense proxy."),
    "RKLB": ("defense", "mapped", "Rocket Lab is mapped as an aerospace and defense proxy."),
    "UUUU": ("uranium_energy", "mapped", "Energy Fuels is a uranium-energy holding."),
    "BRK-B": ("broad_market", "mapped", "Berkshire is mapped as a broad-market diversified proxy."),
    "ETN": ("AI_infrastructure", "mapped", "Eaton is mapped as an electrification and AI-infrastructure proxy."),
    "ABBN.SW": ("Japan_Taiwan_ETF_or_non_US", "mapped", "ABB is a non-US industrial holding."),
    "IART.SW": ("Japan_Taiwan_ETF_or_non_US", "mapped", "Implenia is a non-US holding."),
    "SEMI.AS": ("semiconductor", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "3587.TWO": ("semiconductor", "mapped", "This Taiwan listing is mapped as a semiconductor proxy."),
    "00935.TW": ("semiconductor", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "SMSD.IL": ("semiconductor", "mapped", "This London listing is mapped as a semiconductor proxy."),
    "WTAI.L": ("AI_infrastructure", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "200A.T": ("Japan_Taiwan_ETF_or_non_US", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "2854.T": ("Japan_Taiwan_ETF_or_non_US", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "QQQ": ("mega_cap_tech", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "XLK": ("mega_cap_tech", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "SMH": ("semiconductor", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "SOXX": ("semiconductor", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
    "SPY": ("broad_market", "proxy_only", "ETF proxy only; no holdings lookthrough is used."),
}


def build_portfolio_crowding_diagnostics(
    frames: dict[str, pd.DataFrame],
    tickers: list[str] | tuple[str, ...],
    *,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
    min_samples: int = MIN_CORRELATION_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build research-only portfolio correlation and factor-proxy tables."""

    clean_tickers = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
    pairwise = build_portfolio_correlation_diagnostics(
        frames,
        clean_tickers,
        rolling_window_days=rolling_window_days,
        min_samples=min_samples,
    )
    summary = build_portfolio_crowding_summary(pairwise, clean_tickers, rolling_window_days=rolling_window_days)
    exposure = build_portfolio_factor_proxy_exposure(clean_tickers)
    factor_summary = build_portfolio_factor_crowding_summary(exposure)
    return pairwise, summary, exposure, factor_summary


def build_portfolio_correlation_diagnostics(
    frames: dict[str, pd.DataFrame],
    tickers: tuple[str, ...],
    *,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
    min_samples: int = MIN_CORRELATION_SAMPLES,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    returns = {ticker: _daily_returns(frames.get(ticker)) for ticker in tickers}
    for ticker_a, ticker_b in combinations(tickers, 2):
        series_a = returns.get(ticker_a)
        series_b = returns.get(ticker_b)
        if series_a is None or series_b is None:
            rows.append(_pair_row(ticker_a, ticker_b, 0, rolling_window_days, None, "unavailable", "Price history was unavailable for one or both tickers."))
            continue
        paired = pd.concat([series_a, series_b], axis=1, join="inner").dropna().tail(rolling_window_days)
        sample_count = int(len(paired))
        if sample_count < min_samples:
            rows.append(_pair_row(ticker_a, ticker_b, sample_count, rolling_window_days, None, "insufficient_sample", "Not enough overlapping daily returns for a correlation estimate."))
            continue
        correlation = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
        classification = _correlation_classification(correlation)
        rows.append(
            _pair_row(
                ticker_a,
                ticker_b,
                sample_count,
                rolling_window_days,
                correlation,
                classification,
                _correlation_reason(classification),
            )
        )
    return pd.DataFrame(rows, columns=[
        "ticker_a",
        "ticker_b",
        "sample_count",
        "rolling_window_days",
        "correlation",
        "correlation_bucket",
        "classification",
        "reason",
    ])


def build_portfolio_crowding_summary(
    pairwise: pd.DataFrame,
    tickers: tuple[str, ...],
    *,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    ticker_count = len(tickers)
    usable = pairwise[pd.to_numeric(pairwise.get("correlation"), errors="coerce").notna()].copy()
    high = pairwise[pairwise.get("classification", pd.Series(dtype=str)).astype(str) == "high_overlap"]
    largest_cluster = _largest_high_correlation_cluster(high, tickers)
    avg_corr = float(usable["correlation"].mean()) if not usable.empty else None
    max_corr = float(usable["correlation"].max()) if not usable.empty else None
    sample_count = int(usable["sample_count"].min()) if not usable.empty else 0

    if ticker_count < 2:
        classification = "unavailable"
        reason = "At least two tickers are required. Equal-weight proxy only; no actual weights were supplied."
    elif usable.empty:
        classification = "insufficient_sample" if not pairwise.empty else "unavailable"
        reason = "No pair had enough overlapping returns. Equal-weight proxy only; no actual weights were supplied."
    else:
        common_high_pairs = len(high) >= max(2, ceil(ticker_count / 3))
        if (
            (avg_corr is not None and avg_corr >= HIGH_AVERAGE_CORRELATION)
            or largest_cluster >= ceil(ticker_count * 0.5)
            or common_high_pairs
        ):
            classification = "high_crowding"
        elif (avg_corr is not None and avg_corr >= MODERATE_AVERAGE_CORRELATION) or len(high) > 0:
            classification = "moderate_crowding"
        else:
            classification = "low_crowding"
        reason = _crowding_reason(classification)

    return pd.DataFrame(
        [
            {
                "diagnostic": "portfolio_correlation_crowding",
                "ticker_count": ticker_count,
                "sample_count": sample_count,
                "high_overlap_pair_count": int(len(high)),
                "average_pairwise_correlation": avg_corr,
                "max_pairwise_correlation": max_corr,
                "largest_cluster_size": int(largest_cluster),
                "classification": classification,
                "reason": reason,
            }
        ]
    )


def build_portfolio_factor_proxy_exposure(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        proxy_group, classification, reason = FACTOR_PROXY_MAP.get(
            ticker,
            ("unknown", "unknown", "Ticker is not in the static proxy map."),
        )
        rows.append(
            {
                "ticker": ticker,
                "proxy_group": proxy_group,
                "proxy_source": "static_ticker_proxy" if classification == "mapped" else ("static_etf_proxy" if classification == "proxy_only" else "unmapped"),
                "classification": classification,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=["ticker", "proxy_group", "proxy_source", "classification", "reason"])


def build_portfolio_factor_crowding_summary(exposure: pd.DataFrame) -> pd.DataFrame:
    if exposure.empty:
        return pd.DataFrame(columns=["proxy_group", "ticker_count", "tickers", "classification", "reason"])
    total = len(exposure)
    rows = []
    for proxy_group, group in exposure.groupby("proxy_group", sort=True):
        tickers = group["ticker"].astype(str).tolist()
        has_etf_proxy = group["classification"].astype(str).eq("proxy_only").any()
        has_single_proxy = group["classification"].astype(str).eq("mapped").any()
        share = len(group) / total if total else 0.0
        if proxy_group == "unknown":
            classification = "unknown"
            reason = "These tickers are not mapped; do not infer factor exposure from them."
        elif share >= 0.40 or (has_etf_proxy and has_single_proxy):
            classification = "crowded"
            reason = "This proxy group is a large share of tickers or mixes ETF proxies with overlapping single-name proxies."
        elif share >= 0.25:
            classification = "watch"
            reason = "This proxy group is a meaningful share of tickers."
        else:
            classification = "low"
            reason = "This proxy group is below the warning threshold."
        rows.append(
            {
                "proxy_group": proxy_group,
                "ticker_count": int(len(group)),
                "tickers": ",".join(tickers),
                "classification": classification,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=["proxy_group", "ticker_count", "tickers", "classification", "reason"])


def _daily_returns(frame: pd.DataFrame | None) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    price_column = "Adj Close" if "Adj Close" in frame else ("Close" if "Close" in frame else None)
    if price_column is None:
        return None
    price = pd.to_numeric(frame[price_column], errors="coerce")
    if "Date" in frame:
        price.index = pd.to_datetime(frame["Date"], errors="coerce")
    return price.sort_index().pct_change()


def _pair_row(
    ticker_a: str,
    ticker_b: str,
    sample_count: int,
    rolling_window_days: int,
    correlation: float | None,
    classification: str,
    reason: str,
) -> dict[str, object]:
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "sample_count": sample_count,
        "rolling_window_days": rolling_window_days,
        "correlation": correlation,
        "correlation_bucket": classification,
        "classification": classification,
        "reason": reason,
    }


def _correlation_classification(correlation: float) -> str:
    if correlation >= HIGH_CORRELATION:
        return "high_overlap"
    if correlation >= MODERATE_CORRELATION:
        return "moderate_overlap"
    return "diversifying"


def _correlation_reason(classification: str) -> str:
    if classification == "high_overlap":
        return "Recent daily returns moved together strongly; diversification may be lower than it looks."
    if classification == "moderate_overlap":
        return "Recent daily returns moved together moderately."
    return "Recent daily returns were not strongly correlated."


def _crowding_reason(classification: str) -> str:
    if classification == "high_crowding":
        return "Equal-weight proxy: correlations suggest the portfolio may be crowded into one moving group."
    if classification == "moderate_crowding":
        return "Equal-weight proxy: correlations show some overlap, but not broad crowding."
    return "Equal-weight proxy: no broad high-correlation pattern was found."


def _largest_high_correlation_cluster(high_pairs: pd.DataFrame, tickers: tuple[str, ...]) -> int:
    if high_pairs.empty:
        return 0
    graph: dict[str, set[str]] = defaultdict(set)
    for _, row in high_pairs.iterrows():
        ticker_a = str(row["ticker_a"])
        ticker_b = str(row["ticker_b"])
        graph[ticker_a].add(ticker_b)
        graph[ticker_b].add(ticker_a)
    largest = 0
    seen: set[str] = set()
    for ticker in tickers:
        if ticker in seen or ticker not in graph:
            continue
        stack = [ticker]
        size = 0
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            size += 1
            stack.extend(graph[current] - seen)
        largest = max(largest, size)
    return largest
