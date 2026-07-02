"""Research-only strategy ROI backtests.

This module compares simple long/cash ticker strategies without changing the
production Decision Cockpit, scoring, allocation, labels, cache, or UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.backtest.metrics import annualized_return
from src.backtest.signals import lag_positions
from src.data.fetch import load_daily_data
from src.decision.config import DecisionConfig, load_decision_config, profile_settings
from src.ml.datasets import assert_no_label_leakage, build_supervised_panel, feature_group_columns
from src.ml.models import fit_classifier, predict_positive_probability
from src.ml.scoring import ml_score
from src.ml.validation import deduplicate_prediction_keys, infer_horizon, prediction_merge_keys
from src.portfolio.allocation import AllocationConfig, allocate_from_scores
from src.research.lab import build_features_for_universe
from src.utils.config import FeatureConfig


BUY_AND_HOLD = "Buy & Hold"
SMA_200 = "200-day moving-average rule"
SMA_50_200 = "50/200-day moving-average cross rule"
SSL_PRODUCTION = "Stock Signal Lab production rule"
SSL_WITHOUT_ML = "Stock Signal Lab production rule without ML"
SSL_WITH_ML = "Stock Signal Lab production rule with ML"
SIMPLE_TREND = "Simple trend-only rule"
SIMPLE_TREND_DRAWDOWN_RISK = "Simple trend + drawdown-risk rule"
SIMPLE_TREND_REDUCED_DEFENSIVENESS = "Simple trend + reduced-defensiveness rule"
STRATEGY_ORDER = (
    BUY_AND_HOLD,
    SMA_200,
    SMA_50_200,
    SSL_PRODUCTION,
    SSL_WITHOUT_ML,
    SSL_WITH_ML,
    SIMPLE_TREND,
    SIMPLE_TREND_DRAWDOWN_RISK,
    SIMPLE_TREND_REDUCED_DEFENSIVENESS,
)


@dataclass(frozen=True)
class SimpleROIBacktestConfig:
    """Small set of assumptions for research-only ROI checks."""

    tickers: tuple[str, ...] = ("XLK", "QQQ", "NVDA", "TSM", "PLTR")
    start: str = "2024-01-01"
    end: str = "2026-06-17"
    benchmark: str = "SPY"
    lookback_start: str = "2021-01-01"
    starting_capital: float = 1_000.0
    transaction_cost_bps: float = 0.0
    model_name: str | None = None
    feature_group: str | None = None
    train_window: int = 504
    test_window: int = 63
    step: int = 63
    embargo: int = 20
    profile_name: str = "Balanced"
    use_cache: bool = True


@dataclass(frozen=True)
class StrategyBacktestResult:
    """One ticker/strategy curve and summary row."""

    ticker: str
    strategy: str
    curve: pd.DataFrame
    summary: dict[str, object]


def buy_and_hold_signal(price: pd.Series) -> pd.Series:
    """Stay fully invested after the first executable day."""

    return pd.Series(1.0, index=price.index)


def moving_average_signal(price: pd.Series, window: int = 200) -> pd.Series:
    """Long when price is above its trailing moving average."""

    average = price.rolling(window=window, min_periods=window).mean()
    return (price > average).astype(float).fillna(0.0)


def moving_average_cross_signal(price: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """Long when the short moving average is above the long moving average."""

    short_average = price.rolling(window=short_window, min_periods=short_window).mean()
    long_average = price.rolling(window=long_window, min_periods=long_window).mean()
    return (short_average > long_average).astype(float).fillna(0.0)


def simple_trend_signal(features: pd.DataFrame) -> pd.Series:
    """Long when price is above both 50d and 200d averages."""

    dist_50 = pd.to_numeric(features.get("dist_ma_50d", pd.Series(index=features.index)), errors="coerce")
    dist_200 = pd.to_numeric(features.get("dist_ma_200d", pd.Series(index=features.index)), errors="coerce")
    return ((dist_50 > 0.0) & (dist_200 > 0.0)).astype(float)


def simple_trend_drawdown_risk_signal(features: pd.DataFrame) -> pd.Series:
    """Trend rule with the existing regime drawdown-risk thresholds."""

    dd_60 = pd.to_numeric(features.get("max_drawdown_60d", pd.Series(index=features.index)), errors="coerce")
    dd_120 = pd.to_numeric(features.get("max_drawdown_120d", pd.Series(index=features.index)), errors="coerce")
    return (simple_trend_signal(features).eq(1.0) & (dd_60 > -0.12) & (dd_120 > -0.20)).astype(float)


def simple_trend_reduced_defensiveness_signal(features: pd.DataFrame) -> pd.Series:
    """Less defensive trend rule: require one positive average and no major 120d drawdown."""

    dist_50 = pd.to_numeric(features.get("dist_ma_50d", pd.Series(index=features.index)), errors="coerce")
    dist_200 = pd.to_numeric(features.get("dist_ma_200d", pd.Series(index=features.index)), errors="coerce")
    dd_120 = pd.to_numeric(features.get("max_drawdown_120d", pd.Series(index=features.index)), errors="coerce")
    return (((dist_50 > 0.0) | (dist_200 > 0.0)) & (dd_120 > -0.20)).astype(float)


def run_single_strategy_backtest(
    ticker: str,
    price: pd.Series,
    raw_signal: pd.Series,
    strategy: str,
    *,
    starting_capital: float = 1_000.0,
    transaction_cost_bps: float = 0.0,
) -> StrategyBacktestResult:
    """Run one long/cash ticker strategy with next-day execution."""

    aligned_price = price.dropna().astype(float)
    raw_position = raw_signal.reindex(aligned_price.index).fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    position = lag_positions(raw_position)
    returns = aligned_price.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    trading_cost = turnover * (transaction_cost_bps / 10_000.0)
    strategy_return = position * returns - trading_cost
    equity = starting_capital * (1.0 + strategy_return).cumprod()
    entry_price = aligned_price.shift(1)
    shares = (equity.shift(1).fillna(starting_capital) * position / entry_price).fillna(0.0)

    curve = pd.DataFrame(
        {
            "price": aligned_price,
            "raw_position": raw_position,
            "position": position,
            "turnover": turnover,
            "strategy_return": strategy_return,
            "strategy_equity": equity,
            "fractional_shares": shares,
        },
        index=aligned_price.index,
    )
    summary = summarize_strategy_curve(ticker, strategy, curve, starting_capital)
    return StrategyBacktestResult(ticker=ticker, strategy=strategy, curve=curve, summary=summary)


def summarize_strategy_curve(
    ticker: str,
    strategy: str,
    curve: pd.DataFrame,
    starting_capital: float,
) -> dict[str, object]:
    """Build the required investment metrics for one strategy."""

    if curve.empty:
        return {
            "ticker": ticker,
            "strategy": strategy,
            "final_value": pd.NA,
            "total_return": pd.NA,
            "annualized_return": pd.NA,
            "max_drawdown": pd.NA,
            "number_of_trades": 0,
            "days_in_market": 0,
            "percent_days_in_market": pd.NA,
            "worst_drawdown_start": pd.NA,
            "worst_drawdown_end": pd.NA,
        }

    equity = curve["strategy_equity"].dropna()
    returns = curve["strategy_return"].dropna()
    drawdown = drawdown_details(equity)
    days_in_market = int((curve["position"] > 0).sum())
    return {
        "ticker": ticker,
        "strategy": strategy,
        "final_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / starting_capital - 1.0),
        "annualized_return": annualized_return(returns),
        "max_drawdown": drawdown["max_drawdown"],
        "number_of_trades": int((curve["turnover"] > 1e-12).sum()),
        "days_in_market": days_in_market,
        "percent_days_in_market": float(days_in_market / len(curve)) if len(curve) else pd.NA,
        "worst_drawdown_start": drawdown["start"],
        "worst_drawdown_end": drawdown["end"],
    }


def drawdown_details(equity: pd.Series) -> dict[str, object]:
    """Return max drawdown and the peak/trough dates."""

    clean = equity.dropna()
    if clean.empty:
        return {"max_drawdown": pd.NA, "start": pd.NA, "end": pd.NA}
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    end = drawdown.idxmin()
    start = clean.loc[:end].idxmax()
    return {"max_drawdown": float(drawdown.loc[end]), "start": start, "end": end}


def run_simple_roi_backtest(config: SimpleROIBacktestConfig | None = None) -> pd.DataFrame:
    """Run the requested six-strategy research-only ROI comparison."""

    cfg = config or SimpleROIBacktestConfig()
    decision_config = load_decision_config()
    model_name = cfg.model_name or decision_config.default_model
    feature_group = cfg.feature_group or decision_config.default_feature_group
    tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in cfg.tickers if ticker.strip()))
    frames = load_backtest_frames(cfg, tickers, decision_config)
    feature_frames = build_features_for_universe(frames, FeatureConfig(), cfg.benchmark)
    ticker_features = {ticker: feature_frames[ticker] for ticker in tickers if ticker in feature_frames}
    benchmark_frame = frames.get(cfg.benchmark)
    if benchmark_frame is None or benchmark_frame.empty:
        raise ValueError(f"No benchmark data available for {cfg.benchmark}")
    supervised = build_supervised_panel(
        ticker_features,
        benchmark_price=benchmark_frame["Adj Close"],
        horizon=decision_config.default_label_horizon,
    )
    columns = feature_group_columns(supervised, feature_group) if not supervised.empty else []
    score_panel = (
        build_historical_score_panel(
            supervised,
            columns,
            model_name=model_name,
            train_window=cfg.train_window,
            test_window=cfg.test_window,
            step=cfg.step,
            embargo=cfg.embargo,
            horizon=decision_config.default_label_horizon,
        )
        if columns
        else pd.DataFrame()
    )

    rows: list[dict[str, object]] = []
    for ticker in tickers:
        features = ticker_features.get(ticker)
        if features is None or features.empty or "Adj Close" not in features:
            continue
        window = features.loc[(features.index >= pd.Timestamp(cfg.start)) & (features.index <= pd.Timestamp(cfg.end))]
        if window.empty:
            continue
        price = window["Adj Close"]
        ticker_scores = score_panel[score_panel["Ticker"] == ticker] if not score_panel.empty else pd.DataFrame()
        signals = {
            BUY_AND_HOLD: buy_and_hold_signal(price),
            SMA_200: moving_average_signal(features["Adj Close"]).reindex(price.index),
            SMA_50_200: moving_average_cross_signal(features["Adj Close"]).reindex(price.index),
            SSL_WITHOUT_ML: stock_signal_lab_without_ml_signal(features).reindex(price.index),
            SSL_WITH_ML: stock_signal_lab_with_ml_signal(
                features,
                ticker_scores,
                decision_config,
                cfg.profile_name,
            ).reindex(price.index),
            SIMPLE_TREND: simple_trend_signal(features).reindex(price.index),
            SIMPLE_TREND_DRAWDOWN_RISK: simple_trend_drawdown_risk_signal(features).reindex(price.index),
            SIMPLE_TREND_REDUCED_DEFENSIVENESS: simple_trend_reduced_defensiveness_signal(features).reindex(price.index),
        }
        signals[SSL_PRODUCTION] = signals[SSL_WITH_ML]
        for strategy in STRATEGY_ORDER:
            result = run_single_strategy_backtest(
                ticker,
                price,
                signals[strategy],
                strategy,
                starting_capital=cfg.starting_capital,
                transaction_cost_bps=cfg.transaction_cost_bps,
            )
            rows.append(result.summary)

    return add_comparison_columns(pd.DataFrame(rows))


def load_backtest_frames(
    config: SimpleROIBacktestConfig,
    tickers: tuple[str, ...],
    decision_config: DecisionConfig,
) -> dict[str, pd.DataFrame]:
    """Load ticker, benchmark, and app relative-strength benchmark frames."""

    symbols = list(dict.fromkeys([*tickers, config.benchmark.upper(), "SPY", "QQQ", decision_config.default_benchmark]))
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames[symbol] = load_daily_data(
            symbol,
            start=config.lookback_start,
            end=config.end,
            use_cache=config.use_cache,
        )
    return frames


def stock_signal_lab_without_ml_signal(features: pd.DataFrame) -> pd.Series:
    """A safe no-ML ablation using only the production regime sizing map."""

    allocation_config = AllocationConfig(max_position_size=1.0, cash_floor=0.0, max_gross_exposure=1.0)
    regimes = features.get("regime", pd.Series(index=features.index, dtype=object))
    return regimes.map(allocation_config.regime_multipliers).fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)


def stock_signal_lab_with_ml_signal(
    features: pd.DataFrame,
    score_panel: pd.DataFrame,
    decision_config: DecisionConfig,
    profile_name: str,
) -> pd.Series:
    """Use production allocation helper, converted to per-ticker exposure."""

    if score_panel.empty:
        return pd.Series(0.0, index=features.index)
    profile = profile_settings(decision_config, profile_name)
    allocation_config = AllocationConfig(
        max_position_size=profile.max_single_position_exposure,
        cash_floor=profile.cash_floor,
        max_gross_exposure=1.0 - profile.cash_floor,
        drawdown_risk_threshold=profile.high_drawdown_risk_threshold,
        moderate_drawdown_risk_threshold=profile.moderate_drawdown_risk_threshold,
    )
    data = score_panel.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    rows: list[dict[str, object]] = []
    for _, row in data.sort_values("Date").iterrows():
        score_row = pd.DataFrame(
            [
                {
                    "Ticker": row["Ticker"],
                    "ML Score": row["ML Score"],
                    "ML Drawdown-Risk Probability": row["ML Drawdown-Risk Probability"],
                    "Rule-Based Regime": row.get("Rule-Based Regime", ""),
                    "volatility_60d": row.get("volatility_60d", pd.NA),
                }
            ]
        )
        allocation = allocate_from_scores(score_row, allocation_config)
        target_weight = float(allocation["target_weight"].iloc[0]) if not allocation.empty else 0.0
        rows.append(
            {
                "Date": row["Date"],
                "position": min(1.0, max(0.0, target_weight / profile.max_single_position_exposure)),
            }
        )
    if not rows:
        return pd.Series(0.0, index=features.index)
    positions = pd.DataFrame(rows).drop_duplicates("Date", keep="last").set_index("Date")["position"].sort_index()
    return positions.reindex(features.index).ffill().fillna(0.0).astype(float)


def build_historical_score_panel(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    *,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    horizon: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score historical rows with models trained only on older labeled rows."""

    assert_no_label_leakage(feature_columns)
    if dataset.empty or not feature_columns:
        return pd.DataFrame()
    out = walk_forward_score_rows(
        dataset,
        feature_columns,
        f"label_outperform_{horizon}d",
        model_name=model_name,
        train_window=train_window,
        test_window=test_window,
        step=step,
        embargo=embargo,
        random_state=random_state,
    )
    risk = walk_forward_score_rows(
        dataset,
        feature_columns,
        f"label_drawdown_risk_{horizon}d",
        model_name=model_name,
        train_window=train_window,
        test_window=test_window,
        step=step,
        embargo=embargo,
        random_state=random_state,
    )
    if out.empty or risk.empty:
        return pd.DataFrame()
    keys = prediction_merge_keys(out, risk)
    merged = deduplicate_prediction_keys(out, keys).merge(
        deduplicate_prediction_keys(risk, keys),
        on=keys,
        suffixes=("_out", "_risk"),
    )
    if merged.empty:
        return pd.DataFrame()
    scored = pd.DataFrame(
        {
            "Date": pd.to_datetime(merged["Date"]),
            "Ticker": merged["Ticker"],
            **({"fold": merged["fold"]} if "fold" in merged else {}),
            "ML Outperformance Probability": merged["probability_out"],
            "ML Drawdown-Risk Probability": merged["probability_risk"],
            "ML Score": ml_score(merged["probability_out"], merged["probability_risk"]),
        }
    )
    context_columns = ["Date", "Ticker", "regime", "volatility_60d"]
    context = dataset[[column for column in context_columns if column in dataset]].drop_duplicates(["Date", "Ticker"])
    scored = scored.merge(context, on=["Date", "Ticker"], how="left")
    return scored.rename(columns={"regime": "Rule-Based Regime"})


def walk_forward_score_rows(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    *,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Walk-forward score rows, including rows whose future label is unavailable."""

    if dataset.empty:
        return pd.DataFrame()
    data = dataset.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    unique_dates = pd.DatetimeIndex(pd.Series(data["Date"]).drop_duplicates().sort_values())
    effective_embargo = max(int(embargo), infer_horizon(label_column))
    active_step = int(step or test_window)
    frames: list[pd.DataFrame] = []
    start = 0
    fold = 1
    while start + train_window + effective_embargo < len(unique_dates):
        train_dates = unique_dates[start : start + train_window]
        test_start = start + train_window + effective_embargo
        test_end = min(test_start + test_window, len(unique_dates))
        test_dates = unique_dates[test_start:test_end]
        if len(test_dates) == 0:
            break
        train = data[data["Date"].isin(train_dates)]
        test = data[data["Date"].isin(test_dates)]
        if not train.empty and not test.empty:
            model = fit_classifier(train, feature_columns, label_column, model_name, random_state=random_state)
            probability = pd.Series(predict_positive_probability(model, test[feature_columns]), index=test.index)
            frames.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "Date": test["Date"],
                        "Ticker": test["Ticker"],
                        "probability": probability,
                    }
                )
            )
        start += active_step
        fold += 1
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Ticker"], keep="last")


def add_comparison_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticker winner/loser and Stock Signal Lab comparison flags."""

    if results.empty:
        return results
    output = results.copy()
    strategy_rank = {strategy: index for index, strategy in enumerate(STRATEGY_ORDER)}
    output["_strategy_order"] = output["strategy"].map(strategy_rank).fillna(len(strategy_rank))
    output = output.sort_values(["ticker", "_strategy_order"]).drop(columns=["_strategy_order"]).reset_index(drop=True)
    for ticker, group in output.groupby("ticker", sort=False):
        values = group.set_index("strategy")["final_value"]
        best_strategy = str(values.idxmax()) if not values.empty else ""
        worst_strategy = str(values.idxmin()) if not values.empty else ""
        ssl_value = values.get(SSL_PRODUCTION, pd.NA)
        buy_hold_value = values.get(BUY_AND_HOLD, pd.NA)
        sma_value = values.get(SMA_200, pd.NA)
        without_ml_value = values.get(SSL_WITHOUT_ML, pd.NA)
        with_ml_value = values.get(SSL_WITH_ML, pd.NA)
        ssl_row = group[group["strategy"] == SSL_PRODUCTION].head(1)
        mask = output["ticker"] == ticker
        output.loc[mask, "best_strategy_per_ticker"] = best_strategy
        output.loc[mask, "worst_strategy_per_ticker"] = worst_strategy
        output.loc[mask, "beats_buy_hold"] = output.loc[mask, "final_value"].map(
            lambda value: _better(value, buy_hold_value)
        )
        output.loc[mask, "beats_200dma"] = output.loc[mask, "final_value"].map(lambda value: _better(value, sma_value))
        output.loc[mask, "improves_return_drawdown_tradeoff_vs_ssl"] = output.loc[mask].apply(
            lambda row: _improves_return_drawdown_tradeoff(row, ssl_row.iloc[0]) if not ssl_row.empty else pd.NA,
            axis=1,
        )
        output.loc[mask, "ssl_beats_buy_hold"] = bool(ssl_value > buy_hold_value) if pd.notna(ssl_value) else pd.NA
        output.loc[mask, "ssl_beats_200dma"] = bool(ssl_value > sma_value) if pd.notna(ssl_value) else pd.NA
        output.loc[mask, "ml_adds_value_vs_non_ml_app_rule"] = (
            bool(with_ml_value > without_ml_value) if pd.notna(with_ml_value) and pd.notna(without_ml_value) else pd.NA
        )
    return output


def _better(candidate: object, baseline: object) -> object:
    if pd.isna(candidate) or pd.isna(baseline):
        return pd.NA
    return bool(float(candidate) > float(baseline))


def _return_drawdown_tradeoff(row: pd.Series) -> float | None:
    total_return = row.get("total_return", pd.NA)
    max_drawdown = row.get("max_drawdown", pd.NA)
    if pd.isna(total_return) or pd.isna(max_drawdown):
        return None
    drawdown = abs(float(max_drawdown))
    if drawdown <= 1e-12:
        return float("inf") if float(total_return) > 0 else float(total_return)
    return float(total_return) / drawdown


def _improves_return_drawdown_tradeoff(candidate: pd.Series, baseline: pd.Series) -> object:
    candidate_score = _return_drawdown_tradeoff(candidate)
    baseline_score = _return_drawdown_tradeoff(baseline)
    if candidate_score is None or baseline_score is None:
        return pd.NA
    return bool(float(candidate["total_return"]) > float(baseline["total_return"]) and candidate_score > baseline_score)


def default_lookback_start(start: str, years: int = 3) -> str:
    """Return a conservative data start for features and ML training windows."""

    parsed = pd.Timestamp(start).date()
    return (parsed - timedelta(days=365 * years)).isoformat()


def write_optional_csv(results: pd.DataFrame, output: Path | None) -> None:
    """Write a CSV only when the caller explicitly asks for one."""

    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
