from __future__ import annotations

import math

import pandas as pd

from src.ml.datasets import feature_group_columns
from src.ml.diagnostics import (
    build_feature_family_summary,
    build_feature_quantile_signal_summary,
    build_feature_signal_table,
    build_ml_feature_audit,
    build_ml_feature_signal_diagnostics,
)
from src.ml.feature_selection import (
    FEATURE_REDUNDANCY_REPORT_COLUMNS,
    prune_redundant_features,
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
        "fourier_cycle_clarity",
        "fourier_cycle_strength",
        "fourier_noise_diffusion",
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
        "wavelet_clean_trend",
        "wavelet_trend_quality",
        "wavelet_noise_pressure",
        "wavelet_medium_long_energy_share",
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
        "fourier_cycle_clarity",
        "wavelet_clean_trend",
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


def test_signal_feature_groups_include_derived_transform_features() -> None:
    panel = feature_selection_panel()

    technical = feature_group_columns(panel, "technical")
    technical_fourier = feature_group_columns(panel, "technical_fourier")
    technical_wavelet = feature_group_columns(panel, "technical_wavelet")
    all_columns = feature_group_columns(panel, "all")

    assert "fourier_cycle_clarity" not in technical
    assert "wavelet_clean_trend" not in technical
    assert "fourier_cycle_clarity" in technical_fourier
    assert "wavelet_clean_trend" in technical_wavelet
    assert {"fourier_cycle_clarity", "wavelet_clean_trend"} <= set(all_columns)


def test_redundant_transform_features_are_detected_and_dropped() -> None:
    panel = feature_selection_panel()

    kept, dropped, report = prune_redundant_features(
        panel,
        ["fourier_cycle_clarity", "fourier_amp_1"],
    )

    assert kept == ["fourier_cycle_clarity"]
    assert dropped == ["fourier_amp_1"]
    assert report.loc[0, "dropped_feature"] == "fourier_amp_1"
    assert report.loc[0, "kept_feature"] == "fourier_cycle_clarity"
    assert report.loc[0, "abs_correlation"] == 1.0


def test_redundancy_preference_keeps_derived_signal_over_low_level_transform() -> None:
    panel = feature_selection_panel()

    kept, dropped, _ = prune_redundant_features(
        panel,
        ["wavelet_energy_scale_1", "wavelet_clean_trend"],
    )

    assert kept == ["wavelet_clean_trend"]
    assert dropped == ["wavelet_energy_scale_1"]


def test_redundancy_report_collapses_kept_feature_chains_to_final_feature() -> None:
    rows = 60
    base = pd.Series(range(rows), dtype=float)
    panel = pd.DataFrame(
        {
            "fourier_amp_1": base * 2.0,
            "fourier_energy_concentration": base,
            "fourier_cycle_clarity": base + (base % 3) * 0.01,
        }
    )

    kept, dropped, report = prune_redundant_features(
        panel,
        [
            "fourier_amp_1",
            "fourier_energy_concentration",
            "fourier_cycle_clarity",
        ],
    )

    assert kept == ["fourier_cycle_clarity"]
    assert dropped == ["fourier_amp_1", "fourier_energy_concentration"]
    kept_by_dropped = dict(zip(report["dropped_feature"], report["kept_feature"], strict=False))
    assert kept_by_dropped["fourier_amp_1"] == "fourier_cycle_clarity"
    assert kept_by_dropped["fourier_energy_concentration"] == "fourier_cycle_clarity"


def test_technical_feature_group_is_unchanged_by_transform_pruning() -> None:
    panel = feature_selection_panel()

    assert feature_group_columns(panel, "technical") == feature_group_columns(
        panel,
        "technical",
        prune_redundant_complex=False,
    )


def test_transform_feature_groups_keep_at_least_one_signal_after_pruning() -> None:
    panel = feature_selection_panel()

    for group in ("technical_fourier", "technical_wavelet", "all"):
        columns = feature_group_columns(panel, group)
        assert any(column.startswith(("fourier_", "wavelet_")) for column in columns)


def test_redundancy_report_has_required_columns_and_finite_correlations() -> None:
    panel = feature_selection_panel()
    panel.loc[0, "fourier_amp_1"] = float("inf")

    _, _, report = prune_redundant_features(
        panel,
        ["fourier_cycle_clarity", "fourier_amp_1"],
    )

    assert list(report.columns) == FEATURE_REDUNDANCY_REPORT_COLUMNS
    assert report["abs_correlation"].map(math.isfinite).all()


def test_redundancy_selection_is_deterministic() -> None:
    panel = feature_selection_panel()
    candidates = [
        "fourier_energy_concentration",
        "fourier_cycle_clarity",
        "fourier_amp_1",
        "wavelet_clean_trend",
        "wavelet_energy_scale_1",
    ]

    first = prune_redundant_features(panel, candidates)
    second = prune_redundant_features(panel, candidates)

    assert first[0] == second[0]
    assert first[1] == second[1]
    pd.testing.assert_frame_equal(first[2], second[2])


def test_feature_audit_defaults_to_selected_model_features() -> None:
    audit = build_ml_feature_audit(feature_selection_panel())

    assert audit.inventory_summary.loc[0, "feature_count"] == 15
    assert audit.redundancy_selection_summary.loc[0, "dropped_feature_count"] == 11
    assert set(FEATURE_REDUNDANCY_REPORT_COLUMNS) <= set(audit.redundancy_selection_report.columns)


def test_feature_signal_diagnostics_default_to_selected_model_features() -> None:
    diagnostics = build_ml_feature_signal_diagnostics(feature_selection_panel())

    assert PRUNED_FEATURES.isdisjoint(set(diagnostics.signal_table["feature"]))
    assert PRUNED_FEATURES.isdisjoint(set(diagnostics.quantile_summary["feature"]))


def test_feature_signal_tables_default_to_selected_model_features() -> None:
    panel = feature_selection_panel()

    signal_table = build_feature_signal_table(panel, max_features=None)
    quantile_summary = build_feature_quantile_signal_summary(panel, max_features=50)

    assert len(signal_table) == 15
    assert PRUNED_FEATURES.isdisjoint(set(signal_table["feature"]))
    assert PRUNED_FEATURES.isdisjoint(set(quantile_summary["feature"]))
