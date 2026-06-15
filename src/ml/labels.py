"""Supervised label generation for forward-looking research targets.

Labels intentionally use future data. They should be joined to feature rows only
as targets and must never be included in model input columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.regime import DISTRIBUTION, DOWNTREND_HIGH_RISK

DEFAULT_DRAWDOWN_PENALTY_WEIGHT = 0.5


def forward_return(price: pd.Series, horizon: int = 20) -> pd.Series:
    """Return from date `t` to `t + horizon`."""

    return price.shift(-horizon) / price - 1.0


def forward_excess_return(
    price: pd.Series,
    benchmark_price: pd.Series,
    horizon: int = 20,
) -> pd.Series:
    """Future ticker return minus future benchmark return."""

    aligned_benchmark = benchmark_price.reindex(price.index).ffill()
    return forward_return(price, horizon) - forward_return(aligned_benchmark, horizon)


def forward_drawdown(price: pd.Series, horizon: int = 20) -> pd.Series:
    """Worst forward return from date `t` over the next `horizon` sessions."""

    future_returns = [price.shift(-step) / price - 1.0 for step in range(1, horizon + 1)]
    output = pd.concat(future_returns, axis=1).min(axis=1)
    output.iloc[-horizon:] = np.nan
    return output


def make_risk_adjusted_relative_forward_target(
    price: pd.Series,
    benchmark_price: pd.Series,
    horizon: int = 20,
    drawdown_penalty_weight: float = DEFAULT_DRAWDOWN_PENALTY_WEIGHT,
) -> pd.Series:
    """Forward excess return minus a penalty for worst forward drawdown."""

    excess = forward_excess_return(price.astype(float), benchmark_price.astype(float), horizon)
    drawdown = forward_drawdown(price.astype(float), horizon)
    drawdown_penalty = drawdown.clip(upper=0.0).abs()
    target = excess - drawdown_penalty_weight * drawdown_penalty
    target.name = f"forward_{horizon}d_risk_adjusted_excess_return"
    return target


def forward_regime_deterioration(
    regimes: pd.Series,
    horizon: int = 20,
    bad_regimes: tuple[str, ...] = (DISTRIBUTION, DOWNTREND_HIGH_RISK),
) -> pd.Series:
    """Label whether a bad rule-based regime appears in the next horizon."""

    values: list[float] = []
    bad = set(bad_regimes)
    for pos in range(len(regimes)):
        future = regimes.iloc[pos + 1 : pos + horizon + 1]
        if len(future) < horizon:
            values.append(np.nan)
        else:
            values.append(float(future.isin(bad).any()))
    return pd.Series(values, index=regimes.index, name=f"label_regime_deterioration_{horizon}d")


def make_forward_labels(
    features: pd.DataFrame,
    benchmark_price: pd.Series,
    horizon: int = 20,
    outperformance_threshold: float = 0.02,
    drawdown_threshold: float = -0.10,
) -> pd.DataFrame:
    """Create the Stage 2 classification labels and forward outcome columns."""

    price = features["Adj Close"].astype(float)
    benchmark = benchmark_price.reindex(features.index).ffill().astype(float)
    ticker_forward = forward_return(price, horizon)
    benchmark_forward = forward_return(benchmark, horizon)
    excess = ticker_forward - benchmark_forward
    drawdown = forward_drawdown(price, horizon)
    risk_adjusted = make_risk_adjusted_relative_forward_target(
        price,
        benchmark,
        horizon=horizon,
        drawdown_penalty_weight=DEFAULT_DRAWDOWN_PENALTY_WEIGHT,
    )

    labels = pd.DataFrame(index=features.index)
    labels[f"forward_{horizon}d_return"] = ticker_forward
    labels[f"benchmark_forward_{horizon}d_return"] = benchmark_forward
    labels[f"forward_{horizon}d_excess_return"] = excess
    labels[f"forward_{horizon}d_drawdown"] = drawdown
    labels[f"forward_{horizon}d_risk_adjusted_excess_return"] = risk_adjusted
    labels[f"label_outperform_{horizon}d"] = (excess > outperformance_threshold).astype(float)
    labels[f"label_risk_adjusted_outperform_{horizon}d"] = (
        risk_adjusted > outperformance_threshold
    ).astype(float)
    labels[f"label_drawdown_risk_{horizon}d"] = (drawdown < drawdown_threshold).astype(float)

    invalid = ticker_forward.isna() | benchmark_forward.isna()
    labels.loc[invalid, [f"label_outperform_{horizon}d"]] = np.nan
    labels.loc[risk_adjusted.isna(), [f"label_risk_adjusted_outperform_{horizon}d"]] = np.nan
    labels.loc[drawdown.isna(), [f"label_drawdown_risk_{horizon}d"]] = np.nan

    if "regime" in features:
        labels[f"label_regime_deterioration_{horizon}d"] = forward_regime_deterioration(
            features["regime"], horizon=horizon
        )
    return labels
