"""Price-only AI five-layer rotation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import pandas as pd


AI_LAYER_BASKETS = {
    "Energy": ("CEG", "VST", "NRG"),
    "Chips": ("NVDA", "AVGO", "TSM", "MU"),
    "Infrastructure": ("VRT", "ETN", "ANET", "DELL"),
    "Models/Hyperscalers": ("MSFT", "GOOGL", "META", "AMZN"),
    "Applications": ("PLTR", "NOW", "DDOG", "CRM"),
}
AI_LAYER_TICKERS = tuple(dict.fromkeys(ticker for basket in AI_LAYER_BASKETS.values() for ticker in basket))
AI_LAYER_WINDOWS = (1, 5, 20, 60)
AI_LAYER_DETAIL_COLUMNS = [
    "as_of",
    "layer",
    "tickers",
    "window",
    "available_constituents",
    "constituent_count",
    "coverage",
    "basket_return",
    "qqq_return",
    "relative_qqq_return",
    "breadth",
    "annualized_volatility",
    "max_drawdown",
]
AI_LAYER_SUMMARY_COLUMNS = [
    "as_of",
    "signal_window",
    "market_state",
    "classification",
    "leading_layer",
    "lagging_layer",
    "rotation_direction",
    "rotation_strength",
    "strength_label",
    "positive_layer_count",
    "negative_layer_count",
    "available_layer_count",
    "all_layers_weaker",
    "evidence",
]
AI_LAYER_PERCENT_COLUMNS = frozenset(
    {
        "coverage",
        "basket_return",
        "qqq_return",
        "relative_qqq_return",
        "breadth",
        "annualized_volatility",
        "max_drawdown",
        "rotation_strength",
    }
)


@dataclass(frozen=True)
class AILayerRotationDiagnostics:
    detail: pd.DataFrame
    summary: pd.DataFrame


def build_ai_layer_rotation_diagnostics(
    frames: dict[str, pd.DataFrame],
    *,
    benchmark: str = "QQQ",
    windows: tuple[int, ...] = AI_LAYER_WINDOWS,
    signal_window: int = 5,
) -> AILayerRotationDiagnostics:
    """Build current equal-weight layer metrics and a compact rotation readout."""

    benchmark_price = _price(frames.get(benchmark.upper()))
    if benchmark_price.empty or signal_window not in windows:
        detail = pd.DataFrame(columns=AI_LAYER_DETAIL_COLUMNS)
        return AILayerRotationDiagnostics(
            detail=detail,
            summary=summarize_ai_layer_rotation(detail, signal_window=signal_window),
        )

    as_of = benchmark_price.index[-1]
    rows = []
    for layer, tickers in AI_LAYER_BASKETS.items():
        prices = {ticker: _price(frames.get(ticker)) for ticker in tickers}
        for window in windows:
            rows.append(
                _layer_window_metrics(
                    layer,
                    tickers,
                    prices,
                    benchmark_price,
                    as_of=as_of,
                    window=window,
                )
            )
    detail = pd.DataFrame(rows, columns=AI_LAYER_DETAIL_COLUMNS)
    summary = summarize_ai_layer_rotation(detail, signal_window=signal_window)
    return AILayerRotationDiagnostics(detail=detail, summary=summary)


def summarize_ai_layer_rotation(detail: pd.DataFrame, *, signal_window: int = 5) -> pd.DataFrame:
    """Classify current leadership using absolute returns over one signal window."""

    if detail.empty or not {"layer", "window", "basket_return", "relative_qqq_return"}.issubset(detail.columns):
        return _insufficient_summary(signal_window=signal_window)

    current = detail[detail["window"] == signal_window].copy()
    current["basket_return"] = pd.to_numeric(current["basket_return"], errors="coerce")
    current["relative_qqq_return"] = pd.to_numeric(current["relative_qqq_return"], errors="coerce")
    current = current.dropna(subset=["basket_return", "relative_qqq_return"])
    current = current[current["layer"].isin(AI_LAYER_BASKETS)].drop_duplicates("layer", keep="last")
    available_count = int(len(current))
    as_of = _latest_as_of(current)
    if set(current["layer"]) != set(AI_LAYER_BASKETS):
        return _insufficient_summary(
            signal_window=signal_window,
            as_of=as_of,
            available_layer_count=available_count,
        )

    positive_count = int(current["basket_return"].gt(0).sum())
    negative_count = int(current["basket_return"].lt(0).sum())
    all_layers_weaker = negative_count == len(AI_LAYER_BASKETS)
    leader = current.loc[current["relative_qqq_return"].idxmax()]
    laggard = current.loc[current["relative_qqq_return"].idxmin()]
    strength = float(leader["relative_qqq_return"] - laggard["relative_qqq_return"])

    if all_layers_weaker:
        market_state = "AI risk-off"
        classification = "broad de-risking"
        direction = "All layers lower"
        evidence = f"All five AI layers have negative {signal_window}d basket returns."
    elif positive_count and negative_count:
        market_state = "Rotation"
        classification = "crowded rotation" if positive_count == 1 else "healthy rotation"
        direction = f"{laggard['layer']} → {leader['layer']}"
        evidence = (
            f"{positive_count} layer(s) are higher and {negative_count} are lower over {signal_window}d; "
            "gains are concentrated in one layer."
            if positive_count == 1
            else f"{positive_count} layers are higher and {negative_count} are lower over {signal_window}d."
        )
    elif positive_count == len(AI_LAYER_BASKETS):
        market_state = "Broad advance"
        classification = "healthy rotation"
        direction = "All layers higher"
        evidence = f"All five AI layers have positive {signal_window}d basket returns."
    else:
        market_state = "Mixed"
        classification = "crowded rotation"
        direction = "No clear rotation"
        evidence = f"Layer signs are flat or inconclusive over {signal_window}d."

    row = {
        "as_of": as_of,
        "signal_window": f"{signal_window}d",
        "market_state": market_state,
        "classification": classification,
        "leading_layer": str(leader["layer"]),
        "lagging_layer": str(laggard["layer"]),
        "rotation_direction": direction,
        "rotation_strength": strength,
        "strength_label": _strength_label(strength),
        "positive_layer_count": positive_count,
        "negative_layer_count": negative_count,
        "available_layer_count": available_count,
        "all_layers_weaker": all_layers_weaker,
        "evidence": evidence,
    }
    return pd.DataFrame([row], columns=AI_LAYER_SUMMARY_COLUMNS)


def format_ai_layer_rotation_display(table: pd.DataFrame) -> pd.DataFrame:
    """Format diagnostic percentages without changing export values."""

    output = table.copy()
    for column in output.columns:
        if column in AI_LAYER_PERCENT_COLUMNS:
            output[column] = output[column].map(_pct)
    return output


def _layer_window_metrics(
    layer: str,
    tickers: tuple[str, ...],
    prices: dict[str, pd.Series],
    benchmark_price: pd.Series,
    *,
    as_of: pd.Timestamp,
    window: int,
) -> dict[str, object]:
    base = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "layer": layer,
        "tickers": ",".join(tickers),
        "window": window,
        "available_constituents": "",
        "constituent_count": len(tickers),
        "coverage": 0.0,
        "basket_return": pd.NA,
        "qqq_return": pd.NA,
        "relative_qqq_return": pd.NA,
        "breadth": pd.NA,
        "annualized_volatility": pd.NA,
        "max_drawdown": pd.NA,
    }
    benchmark_window = benchmark_price.loc[:as_of].tail(window + 1)
    if len(benchmark_window) < window + 1:
        return base

    aligned = pd.DataFrame({ticker: price.reindex(benchmark_window.index) for ticker, price in prices.items()})
    available = [
        ticker
        for ticker in tickers
        if aligned[ticker].notna().sum() == window + 1
        and pd.notna(aligned[ticker].iloc[0])
        and pd.notna(aligned[ticker].iloc[-1])
    ]
    base["available_constituents"] = ",".join(available)
    base["coverage"] = len(available) / len(tickers)
    if not available:
        return base

    constituent_prices = aligned[available]
    daily_returns = constituent_prices.pct_change(fill_method=None).iloc[1:]
    basket_daily = daily_returns.mean(axis=1)
    benchmark_daily = benchmark_window.pct_change(fill_method=None).iloc[1:]
    if basket_daily.isna().any() or benchmark_daily.isna().any():
        return base

    basket_return = float((1.0 + basket_daily).prod() - 1.0)
    qqq_return = float((1.0 + benchmark_daily).prod() - 1.0)
    constituent_returns = constituent_prices.iloc[-1] / constituent_prices.iloc[0] - 1.0
    base.update(
        {
            "basket_return": basket_return,
            "qqq_return": qqq_return,
            "relative_qqq_return": basket_return - qqq_return,
            "breadth": float(constituent_returns.gt(0).mean()),
            "annualized_volatility": (
                float(basket_daily.std(ddof=0) * sqrt(252)) if len(basket_daily) > 1 else pd.NA
            ),
            "max_drawdown": _max_drawdown(basket_daily),
        }
    )
    return base


def _price(frame: pd.DataFrame | None) -> pd.Series:
    if frame is None or frame.empty or "Adj Close" not in frame:
        return pd.Series(dtype=float)
    output = pd.to_numeric(frame["Adj Close"], errors="coerce").dropna()
    output.index = pd.to_datetime(output.index)
    return output[~output.index.duplicated(keep="last")].sort_index()


def _max_drawdown(daily_returns: pd.Series) -> float:
    wealth = pd.concat([pd.Series([1.0]), (1.0 + daily_returns).cumprod().reset_index(drop=True)], ignore_index=True)
    return float((wealth / wealth.cummax() - 1.0).min())


def _strength_label(strength: float) -> str:
    if strength < 0.02:
        return "weak"
    if strength < 0.05:
        return "moderate"
    return "strong"


def _latest_as_of(data: pd.DataFrame) -> object:
    if data.empty or "as_of" not in data:
        return pd.NA
    values = data["as_of"].dropna()
    return values.iloc[-1] if not values.empty else pd.NA


def _insufficient_summary(
    *,
    signal_window: int,
    as_of: object = pd.NA,
    available_layer_count: int = 0,
) -> pd.DataFrame:
    row = {
        "as_of": as_of,
        "signal_window": f"{signal_window}d",
        "market_state": "Unavailable",
        "classification": "insufficient_data",
        "leading_layer": pd.NA,
        "lagging_layer": pd.NA,
        "rotation_direction": "Unavailable",
        "rotation_strength": pd.NA,
        "strength_label": "unavailable",
        "positive_layer_count": pd.NA,
        "negative_layer_count": pd.NA,
        "available_layer_count": available_layer_count,
        "all_layers_weaker": pd.NA,
        "evidence": "All five layer baskets need usable price history before rotation is classified.",
    }
    return pd.DataFrame([row], columns=AI_LAYER_SUMMARY_COLUMNS)


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.1%}"
