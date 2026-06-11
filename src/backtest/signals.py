"""Signal helpers for rule-based regime strategies."""

from __future__ import annotations

import pandas as pd

from src.features.regime import DISTRIBUTION, DOWNTREND_HIGH_RISK, SIDEWAYS, UPTREND_HIGH_VOL, UPTREND_LOW_VOL


DEFAULT_REGIME_POSITIONS: dict[str, float] = {
    UPTREND_LOW_VOL: 1.0,
    UPTREND_HIGH_VOL: 1.0,
    SIDEWAYS: 0.25,
    DISTRIBUTION: 0.0,
    DOWNTREND_HIGH_RISK: 0.0,
}


def positions_from_regimes(
    regimes: pd.Series,
    mapping: dict[str, float] | None = None,
) -> pd.Series:
    """Map regime labels to target position weights."""

    active_mapping = mapping or DEFAULT_REGIME_POSITIONS
    return regimes.map(active_mapping).fillna(0.0).astype(float)


def lag_positions(positions: pd.Series, lag: int = 1) -> pd.Series:
    """Shift positions so today's signal cannot earn today's return."""

    return positions.shift(lag).fillna(0.0).clip(lower=0.0, upper=1.0)

