"""Interpretable rule-based regime classification."""

from __future__ import annotations

import pandas as pd


UPTREND_LOW_VOL = "Uptrend / low volatility"
UPTREND_HIGH_VOL = "Uptrend / high volatility"
SIDEWAYS = "Sideways / low conviction"
DISTRIBUTION = "Distribution / weakening trend"
DOWNTREND_HIGH_RISK = "Downtrend / high risk"

REGIME_ORDER = (
    UPTREND_LOW_VOL,
    UPTREND_HIGH_VOL,
    SIDEWAYS,
    DISTRIBUTION,
    DOWNTREND_HIGH_RISK,
)


def _get(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    return default if pd.isna(value) else float(value)


def classify_regime(features: pd.DataFrame, use_signal_features: bool = True) -> pd.DataFrame:
    """Classify each row into an explainable market regime."""

    output = features.copy()
    vol_threshold = output["volatility_60d"].rolling(252, min_periods=60).median()
    if vol_threshold.dropna().empty:
        vol_threshold = pd.Series(0.30, index=output.index)
    else:
        vol_threshold = vol_threshold.fillna(vol_threshold.dropna().median())

    regimes: list[str] = []
    rationales: list[str] = []
    flags: list[str] = []

    for idx, row in output.iterrows():
        ret_60 = _get(row, "return_60d")
        ret_120 = _get(row, "return_120d")
        dist_50 = _get(row, "dist_ma_50d")
        dist_200 = _get(row, "dist_ma_200d")
        vol_60 = _get(row, "volatility_60d")
        dd_60 = _get(row, "max_drawdown_60d")
        dd_120 = _get(row, "max_drawdown_120d")
        rs_spy = _get(row, "rs_spy_60d")
        rs_qqq = _get(row, "rs_qqq_60d")
        volume_z = _get(row, "volume_z_20d")
        entropy = _get(row, "fourier_spectral_entropy", default=0.5)
        noise = _get(row, "wavelet_short_noise_intensity", default=0.0)

        high_vol = vol_60 > float(vol_threshold.loc[idx]) or vol_60 > 0.35
        trend_positive = ret_60 > 0 and ret_120 > 0 and dist_50 > 0 and dist_200 > 0
        relative_positive = rs_spy >= -0.02 or rs_qqq >= -0.02
        major_drawdown = dd_60 < -0.12 or dd_120 < -0.20
        weakening = ret_60 < 0 or dist_50 < 0 or (volume_z > 1.5 and ret_60 < 0.03)

        row_flags: list[str] = []
        if major_drawdown:
            row_flags.append("drawdown")
        if volume_z > 2.0:
            row_flags.append("volume anomaly")
        if use_signal_features and entropy > 0.8:
            row_flags.append("diffuse spectrum")
        if use_signal_features and noise > 0.35:
            row_flags.append("short-scale noise")

        if major_drawdown and (ret_60 < 0 or dist_200 < 0):
            regime = DOWNTREND_HIGH_RISK
            rationale = "large drawdown with negative trend or below 200d average"
        elif trend_positive and relative_positive:
            regime = UPTREND_HIGH_VOL if high_vol else UPTREND_LOW_VOL
            rationale = "positive 60d/120d trend, above key averages, acceptable relative strength"
        elif weakening:
            regime = DISTRIBUTION
            rationale = "weakening trend, below 50d average, or negative volume anomaly"
        else:
            regime = SIDEWAYS
            rationale = "mixed trend and relative-strength evidence"

        regimes.append(regime)
        rationales.append(rationale)
        flags.append(", ".join(row_flags) if row_flags else "")

    output["regime"] = regimes
    output["regime_rationale"] = rationales
    output["risk_flags"] = flags
    return output

