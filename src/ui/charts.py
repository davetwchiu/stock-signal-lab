"""Plotly chart helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def price_chart(features: pd.DataFrame, title: str) -> go.Figure:
    """Price chart with moving averages."""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=features.index, y=features["Adj Close"], name="Adj Close"))
    for column in ("ma_50d", "ma_200d"):
        if column in features:
            fig.add_trace(go.Scatter(x=features.index, y=features[column], name=column))
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def drawdown_chart(price: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Drawdown chart from adjusted close."""

    drawdown = price / price.cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="Drawdown"))
    fig.update_layout(title=title, height=260, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def feature_chart(features: pd.DataFrame) -> go.Figure:
    """Compact Fourier/wavelet summary chart."""

    fig = go.Figure()
    for column in (
        "fourier_energy_concentration",
        "fourier_spectral_entropy",
        "wavelet_short_noise_intensity",
        "wavelet_trend_return",
    ):
        if column in features:
            fig.add_trace(go.Scatter(x=features.index, y=features[column], name=column))
    fig.update_layout(title="Signal-processing feature summaries", height=300, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def equity_curve_chart(curve: pd.DataFrame) -> go.Figure:
    """Strategy, buy-and-hold, and benchmark equity curves."""

    fig = go.Figure()
    for column, name in (
        ("strategy_equity", "Strategy"),
        ("buy_hold_equity", "Buy & Hold"),
        ("benchmark_equity", "Benchmark"),
    ):
        if column in curve:
            fig.add_trace(go.Scatter(x=curve.index, y=curve[column], name=name))
    fig.update_layout(title="Equity curve", height=360, margin=dict(l=20, r=20, t=45, b=20))
    return fig

