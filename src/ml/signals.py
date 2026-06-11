"""ML-assisted signal rules for research backtests."""

from __future__ import annotations

import pandas as pd


def ml_probability_signal(
    outperform_probability: pd.Series,
    drawdown_risk_probability: pd.Series,
    outperform_threshold: float = 0.60,
    drawdown_risk_threshold: float = 0.40,
    reduced_exposure: float = 0.25,
) -> pd.Series:
    """Simple long/reduce/exit rule from advisory probabilities."""

    aligned = pd.concat(
        [
            outperform_probability.rename("outperform_probability"),
            drawdown_risk_probability.rename("drawdown_risk_probability"),
        ],
        axis=1,
    ).dropna()
    signal = pd.Series(0.0, index=aligned.index)
    signal.loc[
        (aligned["outperform_probability"] >= outperform_threshold)
        & (aligned["drawdown_risk_probability"] < drawdown_risk_threshold)
    ] = 1.0
    signal.loc[
        (aligned["outperform_probability"] >= outperform_threshold)
        & (aligned["drawdown_risk_probability"] >= drawdown_risk_threshold)
    ] = reduced_exposure
    return signal

