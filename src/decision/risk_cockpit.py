"""Hard price-data Risk Cockpit helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MARKET_STATES = ("Normal", "Elevated", "Stressed")
THEME_STATES = ("Healthy", "Weakening", "Stressed", "Mixed")
SINGLE_NAME_STATES = (
    "Healthy trend",
    "Normal pullback",
    "Extended, do not chase",
    "Trend damage",
    "Risk reduction candidate",
    "Watch only",
)
RISK_COCKPIT_TICKERS = ("SPY", "QQQ", "SOXX", "SMH", "TLT", "IEF")


@dataclass(frozen=True)
class RiskCockpit:
    market_state: str
    theme_state: str
    market_panel: pd.DataFrame
    theme_panel: pd.DataFrame
    single_name_health: pd.DataFrame
    memo: str


def _price(frame: pd.DataFrame | None) -> pd.Series:
    if frame is None or frame.empty or "Adj Close" not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame["Adj Close"], errors="coerce").dropna().sort_index()


def _last(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(clean.iloc[-1])


def _feature_or_calc(frame: pd.DataFrame | None, column: str, calc: pd.Series) -> float | None:
    if frame is not None and not frame.empty and column in frame:
        value = _last(frame[column])
        if value is not None:
            return value
    return _last(calc)


def _metrics(frames: dict[str, pd.DataFrame], ticker: str) -> dict[str, object]:
    frame = frames.get(ticker)
    price = _price(frame)
    if price.empty:
        return {"Ticker": ticker}

    returns = price.pct_change()
    ma_50 = price.rolling(50, min_periods=50).mean()
    ma_200 = price.rolling(200, min_periods=200).mean()
    vol_20 = returns.rolling(20, min_periods=20).std() * (252**0.5)
    vol_60 = returns.rolling(60, min_periods=60).std() * (252**0.5)
    recent_high = price.rolling(126, min_periods=20).max()

    current_vol_20 = _feature_or_calc(frame, "volatility_20d", vol_20)
    current_vol_60 = _feature_or_calc(frame, "volatility_60d", vol_60)
    vol_expansion = None
    if current_vol_20 is not None and current_vol_60 and current_vol_60 > 0:
        vol_expansion = current_vol_20 / current_vol_60 - 1.0

    return {
        "Ticker": ticker,
        "Date": price.index[-1],
        "Price": float(price.iloc[-1]),
        "20d return": _feature_or_calc(frame, "return_20d", price / price.shift(20) - 1.0),
        "60d return": _feature_or_calc(frame, "return_60d", price / price.shift(60) - 1.0),
        "120d return": _feature_or_calc(frame, "return_120d", price / price.shift(120) - 1.0),
        "Vs 50DMA": _feature_or_calc(frame, "dist_ma_50d", price / ma_50 - 1.0),
        "Vs 200DMA": _feature_or_calc(frame, "dist_ma_200d", price / ma_200 - 1.0),
        "60d drawdown": _feature_or_calc(
            frame,
            "max_drawdown_60d",
            price / price.rolling(60, min_periods=20).max() - 1.0,
        ),
        "120d drawdown": _feature_or_calc(
            frame,
            "max_drawdown_120d",
            price / price.rolling(120, min_periods=20).max() - 1.0,
        ),
        "From 126d high": _last(price / recent_high - 1.0),
        "20d vol": current_vol_20,
        "60d vol": current_vol_60,
        "Vol expansion": vol_expansion,
    }


def _relative_return(frames: dict[str, pd.DataFrame], ticker: str, benchmark: str, window: int = 60) -> float | None:
    lhs = _price(frames.get(ticker))
    rhs = _price(frames.get(benchmark))
    if len(lhs) <= window or len(rhs) <= window:
        return None
    aligned = pd.concat([lhs, rhs], axis=1, join="inner").dropna()
    if len(aligned) <= window:
        return None
    ticker_return = aligned.iloc[-1, 0] / aligned.iloc[-window - 1, 0] - 1.0
    benchmark_return = aligned.iloc[-1, 1] / aligned.iloc[-window - 1, 1] - 1.0
    return float(ticker_return - benchmark_return)


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def build_market_stress_panel(frames: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    qqq = _metrics(frames, "QQQ")
    qqq_rs_spy = _relative_return(frames, "QQQ", "SPY")
    tlt = _metrics(frames, "TLT") if "TLT" in frames else _metrics(frames, "IEF")

    if "Price" not in qqq or qqq_rs_spy is None:
        panel = pd.DataFrame(
            [
                {
                    "State": "Elevated",
                    "QQQ 60d RS vs SPY": qqq_rs_spy,
                    "QQQ vs 50DMA": qqq.get("Vs 50DMA"),
                    "QQQ vs 200DMA": qqq.get("Vs 200DMA"),
                    "QQQ from 126d high": qqq.get("From 126d high"),
                    "QQQ 20d return": qqq.get("20d return"),
                    "QQQ 60d return": qqq.get("60d return"),
                    "QQQ vol expansion": qqq.get("Vol expansion"),
                    "TLT/IEF vs 200DMA": tlt.get("Vs 200DMA"),
                    "Evidence": "QQQ/SPY hard price data is incomplete.",
                }
            ]
        )
        return "Elevated", panel

    flags: list[str] = []
    if (qqq.get("Vs 200DMA") or 0) < 0:
        flags.append("QQQ is below its 200DMA")
    if (qqq.get("Vs 50DMA") or 0) < 0 and (qqq.get("20d return") or 0) < 0:
        flags.append("QQQ is below its 50DMA with a negative 20d return")
    if (qqq.get("From 126d high") or 0) <= -0.10:
        flags.append("QQQ is more than 10% below its 126d high")
    if qqq_rs_spy is not None and qqq_rs_spy <= -0.03:
        flags.append("QQQ is lagging SPY over 60d")
    if (qqq.get("Vol expansion") or 0) >= 0.25:
        flags.append("QQQ 20d volatility is expanding versus 60d volatility")

    if (qqq.get("Vs 200DMA") or 0) < 0 or len(flags) >= 3:
        state = "Stressed"
    elif flags:
        state = "Elevated"
    else:
        state = "Normal"

    panel = pd.DataFrame(
        [
            {
                "State": state,
                "QQQ 60d RS vs SPY": qqq_rs_spy,
                "QQQ vs 50DMA": qqq.get("Vs 50DMA"),
                "QQQ vs 200DMA": qqq.get("Vs 200DMA"),
                "QQQ from 126d high": qqq.get("From 126d high"),
                "QQQ 20d return": qqq.get("20d return"),
                "QQQ 60d return": qqq.get("60d return"),
                "QQQ vol expansion": qqq.get("Vol expansion"),
                "TLT/IEF vs 200DMA": tlt.get("Vs 200DMA"),
                "Evidence": (
                    "; ".join(flags)
                    if flags
                    else "QQQ trend, drawdown, relative strength, and volatility are not flashing stress."
                ),
            }
        ]
    )
    return state, panel


def build_theme_stress_panel(frames: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for ticker in ("SOXX", "SMH"):
        metrics = _metrics(frames, ticker)
        rs_qqq = _relative_return(frames, ticker, "QQQ")
        missing = "Price" not in metrics or rs_qqq is None
        weak = (
            missing
            or (metrics.get("Vs 50DMA") or 0) < 0
            or (rs_qqq is not None and rs_qqq <= -0.03)
            or (metrics.get("From 126d high") or 0) <= -0.10
        )
        stressed = (
            (metrics.get("Vs 200DMA") or 0) < 0
            or (rs_qqq is not None and rs_qqq <= -0.08)
            or (metrics.get("From 126d high") or 0) <= -0.18
        )
        rows.append(
            {
                "Ticker": ticker,
                "60d RS vs QQQ": rs_qqq,
                "Vs 50DMA": metrics.get("Vs 50DMA"),
                "Vs 200DMA": metrics.get("Vs 200DMA"),
                "From 126d high": metrics.get("From 126d high"),
                "20d return": metrics.get("20d return"),
                "60d return": metrics.get("60d return"),
                "_missing": missing,
                "_weak": weak,
                "_stressed": stressed,
            }
        )

    weak_count = sum(bool(row["_weak"]) for row in rows)
    stressed_count = sum(bool(row["_stressed"]) for row in rows)
    missing_count = sum(bool(row["_missing"]) for row in rows)
    if stressed_count == 2:
        state = "Stressed"
    elif missing_count:
        state = "Mixed"
    elif weak_count == 0:
        state = "Healthy"
    elif weak_count == 2:
        state = "Weakening"
    else:
        state = "Mixed"

    panel = pd.DataFrame(rows).drop(columns=["_missing", "_weak", "_stressed"])
    panel.insert(0, "State", state)
    return state, panel


def classify_single_name_health(frames: dict[str, pd.DataFrame], ticker: str) -> dict[str, object]:
    metrics = _metrics(frames, ticker)
    rs_qqq = _relative_return(frames, ticker, "QQQ")
    rs_soxx = _relative_return(frames, ticker, "SOXX")
    if "Price" not in metrics:
        state = "Watch only"
    else:
        dist_50 = metrics.get("Vs 50DMA")
        dist_200 = metrics.get("Vs 200DMA")
        ret_20 = metrics.get("20d return")
        ret_60 = metrics.get("60d return")
        high_gap = metrics.get("From 126d high")
        dd_60 = metrics.get("60d drawdown")
        dd_120 = metrics.get("120d drawdown")
        below_50 = dist_50 is not None and dist_50 < 0
        below_200 = dist_200 is not None and dist_200 < 0

        if below_200 and (
            (dd_120 is not None and dd_120 <= -0.20)
            or (ret_60 is not None and ret_60 <= -0.10)
            or (rs_qqq is not None and rs_qqq <= -0.08)
        ):
            state = "Risk reduction candidate"
        elif below_200 or (below_50 and (ret_60 or 0) < 0 and (rs_qqq or 0) < 0):
            state = "Trend damage"
        elif (dist_50 or 0) >= 0.12 and (ret_20 or 0) >= 0.08 and (high_gap or 0) >= -0.05:
            state = "Extended, do not chase"
        elif not below_200 and (below_50 or (dd_60 is not None and -0.12 <= dd_60 <= -0.05)):
            state = "Normal pullback"
        elif (dist_50 or 0) >= 0 and (dist_200 or 0) >= 0 and (ret_60 or 0) >= 0 and (rs_qqq is None or rs_qqq >= -0.02):
            state = "Healthy trend"
        else:
            state = "Watch only"

    metrics.update(
        {
            "State": state,
            "60d RS vs QQQ": rs_qqq,
            "60d RS vs SOXX": rs_soxx,
            "Evidence": (
                f"50DMA {_pct(metrics.get('Vs 50DMA'))}, 200DMA {_pct(metrics.get('Vs 200DMA'))}, "
                f"126d-high gap {_pct(metrics.get('From 126d high'))}, "
                f"60d RS vs QQQ {_pct(rs_qqq)}"
            ),
        }
    )
    return metrics


def build_single_name_trend_health(frames: dict[str, pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    rows = [classify_single_name_health(frames, ticker) for ticker in tickers]
    if not rows:
        return pd.DataFrame()
    columns = [
        "Ticker",
        "State",
        "Price",
        "20d return",
        "60d return",
        "120d return",
        "60d RS vs QQQ",
        "60d RS vs SOXX",
        "Vs 50DMA",
        "Vs 200DMA",
        "From 126d high",
        "60d drawdown",
        "Vol expansion",
        "Evidence",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def build_decision_memo(
    market_state: str,
    theme_state: str,
    market_panel: pd.DataFrame,
    theme_panel: pd.DataFrame,
    single_name_health: pd.DataFrame,
) -> str:
    damaged = 0
    if not single_name_health.empty:
        damaged = int(
            single_name_health["State"]
            .isin(["Trend damage", "Risk reduction candidate", "Watch only"])
            .sum()
        )
    market_evidence = (
        str(market_panel.iloc[0].get("Evidence", ""))
        if not market_panel.empty
        else "Market evidence is unavailable."
    )
    theme_lag = []
    if not theme_panel.empty:
        for _, row in theme_panel.iterrows():
            rs = row.get("60d RS vs QQQ")
            if rs is not None and not pd.isna(rs) and float(rs) < 0:
                theme_lag.append(f"{row.get('Ticker')} 60d relative strength vs QQQ is {_pct(rs)}")
    theme_evidence = (
        "; ".join(theme_lag)
        if theme_lag
        else "SOXX and SMH relative strength versus QQQ is not broadly negative."
    )

    return (
        f"Market stress is {market_state} and AI/semiconductor theme stress is {theme_state}. "
        f"What happened: {market_evidence} {theme_evidence} "
        f"Single-name review shows {damaged} holdings with trend damage, watch-only, or risk-reduction status. "
        "The evidence is hard price data: relative strength, 50DMA/200DMA trend, "
        "distance from recent high, drawdown, volatility expansion, and trailing returns. "
        "For portfolio exposure, this is an exposure-discipline warning when broad or theme stress rises. "
        "It does not claim a buy/sell signal, forecast, price target, ranking change, or sizing instruction. "
        "ML remains audit-only and must not override hard risk evidence."
    )


def build_risk_cockpit(frames: dict[str, pd.DataFrame], tickers: list[str]) -> RiskCockpit:
    market_state, market_panel = build_market_stress_panel(frames)
    theme_state, theme_panel = build_theme_stress_panel(frames)
    single_name_health = build_single_name_trend_health(frames, tickers)
    memo = build_decision_memo(market_state, theme_state, market_panel, theme_panel, single_name_health)
    return RiskCockpit(
        market_state=market_state,
        theme_state=theme_state,
        market_panel=market_panel,
        theme_panel=theme_panel,
        single_name_health=single_name_health,
        memo=memo,
    )
