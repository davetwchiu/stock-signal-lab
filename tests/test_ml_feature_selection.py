from __future__ import annotations

import pandas as pd

from src.ml.datasets import feature_group_columns
from src.ml.diagnostics import (
    build_feature_family_summary,
    build_feature_quantile_signal_summary,
    build_feature_signal_table,
    build_ml_feature_audit,
    build_ml_feature_signal_diagnostics,
)


PRUNED_FEATURES = {
    "daily_return",
    "return_5d",
    "return_120d",
    "rs_spy_120d",
    "rs_qqq_120d",
    "fourier_freq_1",
    "fourier_period_1",
    "fourier_amp_2",
    "fourier_amp_3",
    "wavelet_available",
    "wavelet_energy_scale_2",
    "wavelet_energy_scale_3",
}


def feature_selection_panel() -> pd.DataFrame:
    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "regime",
        "regime_rationale",
        "risk_flags",
        "daily_return",
        "return_5d",
        "return_20d",
        "return_60d",
        "return_120d",
        "volatility_20d",
        "volatility_60d",
        "max_drawdown_60d",
        "max_drawdown_120d",
        "dist_ma_50d",
        "dist_ma_200d",
        "volume_z_20d",
        "volume_z_60d",
        "rsi_14d",
        "rs_spy_60d",
        "rs_spy_120d",
        "rs_qqq_60d",
        "rs_qqq_120d",
        "fourier_energy_concentration",
        "fourier_spectral_entropy",
        "fourier_freq_1",
        "fourier_period_1",
        "fourier_amp_1",
        "fourier_freq_2",
        "fourier_period_2",
        "fourier_amp_2",
        "fourier_freq_3",
        "fourier_period_3",
        "fourier_amp_3",
        "wavelet_available",
        "wavelet_trend_return",
        "wavelet_short_noise_intensity",
        "wavelet_energy_scale_1",
        "wavelet_energy_scale_2",
        "wavelet_energy_scale_3",
        "label_outperform_20d",
        "forward_20d_return",
        "benchmark_forward_20d_return",
    ]
    rows = 40
    return pd.DataFrame({column: [float(row) for row in range(rows)] for column in columns})


def test_all_feature_group_uses_curated_model_features() -> None:
    columns = feature_group_columns(feature_selection_panel(), "all")

    assert columns == [
        "return_20d",
        "return_60d",
        "volatility_20d",
        "volatility_60d",
        "max_drawdown_60d",
        "max_drawdown_120d",
        "dist_ma_50d",
        "dist_ma_200d",
        "volume_z_20d",
        "volume_z_60d",
        "rsi_14d",
        "rs_spy_60d",
        "rs_qqq_60d",
        "fourier_energy_concentration",
        "fourier_spectral_entropy",
        "fourier_amp_1",
        "wavelet_trend_return",
        "wavelet_short_noise_intensity",
        "wavelet_energy_scale_1",
    ]


def test_feature_group_excludes_redundant_or_near_constant_features() -> None:
    columns = feature_group_columns(feature_selection_panel(), "all")

    assert PRUNED_FEATURES.isdisjoint(columns)


def test_curated_feature_group_preserves_broad_families() -> None:
    columns = feature_group_columns(feature_selection_panel(), "all")
    families = set(build_feature_family_summary(columns)["family"])

    assert families >= {
        "momentum / return",
        "trend / moving average",
        "volatility",
        "relative strength / benchmark-relative",
        "volume / liquidity",
        "RSI / technical",
        "Fourier / wavelet / complex transform",
    }
    assert any(column.startswith("fourier_") for column in columns)
    assert any(column.startswith("wavelet_") for column in columns)


def test_feature_audit_defaults_to_selected_model_features() -> None:
    audit = build_ml_feature_audit(feature_selection_panel())

    assert audit.inventory_summary.loc[0, "feature_count"] == 19


def test_feature_signal_diagnostics_default_to_selected_model_features() -> None:
    diagnostics = build_ml_feature_signal_diagnostics(feature_selection_panel())

    assert PRUNED_FEATURES.isdisjoint(set(diagnostics.signal_table["feature"]))
    assert PRUNED_FEATURES.isdisjoint(set(diagnostics.quantile_summary["feature"]))


def test_feature_signal_tables_default_to_selected_model_features() -> None:
    panel = feature_selection_panel()

    signal_table = build_feature_signal_table(panel, max_features=None)
    quantile_summary = build_feature_quantile_signal_summary(panel, max_features=50)

    assert len(signal_table) == 19
    assert PRUNED_FEATURES.isdisjoint(set(signal_table["feature"]))
    assert PRUNED_FEATURES.isdisjoint(set(quantile_summary["feature"]))
