"""Dataset assembly and feature-group selection for Stage 2 ML research."""

from __future__ import annotations

import re

import pandas as pd

from src.ml.feature_selection import prune_redundant_features
from src.ml.labels import make_forward_labels


LABEL_PATTERN = re.compile(r"^(label_|forward_|benchmark_forward_)")
TECHNICAL_PREFIXES = (
    "daily_return",
    "return_",
    "volatility_",
    "max_drawdown_",
    "dist_ma_",
    "volume_z_",
    "rsi_",
    "rs_",
)
BASE_EXCLUDE = {
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "regime",
    "regime_rationale",
    "risk_flags",
}
MODEL_FEATURE_EXCLUDE = {
    "daily_return",
    "return_5d",
    "return_120d",
    "rs_spy_120d",
    "rs_qqq_120d",
    "fourier_freq_1",
    "fourier_period_1",
    "fourier_freq_2",
    "fourier_period_2",
    "fourier_amp_2",
    "fourier_freq_3",
    "fourier_period_3",
    "fourier_amp_3",
    "wavelet_available",
    "wavelet_energy_scale_2",
    "wavelet_energy_scale_3",
}


def _prune_transform_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    kept_columns, _, _ = prune_redundant_features(frame, columns)
    return kept_columns


def feature_group_columns(
    frame: pd.DataFrame,
    group: str = "all",
    *,
    prune_redundant_complex: bool = True,
) -> list[str]:
    """Return numeric feature columns for a named feature group."""

    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    candidates = [
        column
        for column in numeric_columns
        if column not in BASE_EXCLUDE
        and column not in MODEL_FEATURE_EXCLUDE
        and not LABEL_PATTERN.match(column)
        and not frame[column].isna().all()
    ]

    technical = [column for column in candidates if column.startswith(TECHNICAL_PREFIXES)]
    fourier = [column for column in candidates if column.startswith("fourier_")]
    wavelet = [column for column in candidates if column.startswith("wavelet_")]
    if group == "technical":
        return technical

    if prune_redundant_complex:
        fourier = _prune_transform_columns(frame, fourier)
        wavelet = _prune_transform_columns(frame, wavelet)

    if group == "technical_fourier":
        return technical + fourier
    if group == "technical_wavelet":
        return technical + wavelet
    if group == "all":
        return technical + fourier + wavelet
    raise ValueError("group must be one of: technical, technical_fourier, technical_wavelet, all")


def assert_no_label_leakage(feature_columns: list[str]) -> None:
    """Raise if target or forward-outcome columns are present in model features."""

    leaked = [column for column in feature_columns if LABEL_PATTERN.match(column)]
    if leaked:
        raise ValueError(f"Label/forward outcome columns cannot be model features: {leaked}")


def build_supervised_panel(
    feature_frames: dict[str, pd.DataFrame],
    benchmark_price: pd.Series,
    horizon: int = 20,
    outperformance_threshold: float = 0.02,
    drawdown_threshold: float = -0.10,
) -> pd.DataFrame:
    """Create one panel DataFrame with features, labels, ticker, and date."""

    rows: list[pd.DataFrame] = []
    for ticker, frame in feature_frames.items():
        if frame.empty or "Adj Close" not in frame:
            continue
        labels = make_forward_labels(
            frame,
            benchmark_price=benchmark_price,
            horizon=horizon,
            outperformance_threshold=outperformance_threshold,
            drawdown_threshold=drawdown_threshold,
        )
        combined = frame.join(labels)
        combined.insert(0, "Ticker", ticker)
        combined.insert(1, "Date", combined.index)
        rows.append(combined.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    return panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def latest_feature_rows(feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the latest feature row for each ticker."""

    rows: list[pd.DataFrame] = []
    for ticker, frame in feature_frames.items():
        if frame.empty:
            continue
        latest = frame.iloc[[-1]].copy()
        latest.insert(0, "Ticker", ticker)
        latest.insert(1, "Date", latest.index)
        rows.append(latest.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
