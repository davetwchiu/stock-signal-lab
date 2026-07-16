from __future__ import annotations

import pandas as pd
import pytest

from src.research.ai_layer_rotation import (
    AI_LAYER_BASKETS,
    AI_LAYER_DETAIL_COLUMNS,
    AI_LAYER_SUMMARY_COLUMNS,
    build_ai_layer_rotation_diagnostics,
    format_ai_layer_rotation_display,
    summarize_ai_layer_rotation,
)
from src.research.export import export_research_lab_diagnostics


def _price_frame(daily_return: float, rows: int = 81) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=rows, freq="B")
    price = 100.0 * (1.0 + pd.Series(daily_return, index=dates)).cumprod()
    return pd.DataFrame({"Adj Close": price}, index=dates)


def _rotation_frames() -> dict[str, pd.DataFrame]:
    layer_returns = {
        "Energy": 0.004,
        "Chips": 0.003,
        "Infrastructure": 0.002,
        "Models/Hyperscalers": -0.001,
        "Applications": -0.002,
    }
    frames = {"QQQ": _price_frame(0.001)}
    for layer, tickers in AI_LAYER_BASKETS.items():
        for ticker in tickers:
            frames[ticker] = _price_frame(layer_returns[layer])
    return frames


def _summary_detail(returns: list[float]) -> pd.DataFrame:
    relative = [value - 0.001 for value in returns]
    return pd.DataFrame(
        {
            "as_of": ["2026-07-15"] * 5,
            "layer": list(AI_LAYER_BASKETS),
            "window": [5] * 5,
            "basket_return": returns,
            "relative_qqq_return": relative,
        }
    )


def test_builds_all_layer_windows_and_price_metrics() -> None:
    diagnostics = build_ai_layer_rotation_diagnostics(_rotation_frames())

    assert list(diagnostics.detail.columns) == AI_LAYER_DETAIL_COLUMNS
    assert list(diagnostics.summary.columns) == AI_LAYER_SUMMARY_COLUMNS
    assert len(diagnostics.detail) == 20
    assert set(diagnostics.detail["layer"]) == set(AI_LAYER_BASKETS)
    assert set(diagnostics.detail["window"]) == {1, 5, 20, 60}

    energy_5d = diagnostics.detail[
        (diagnostics.detail["layer"] == "Energy") & (diagnostics.detail["window"] == 5)
    ].iloc[0]
    assert energy_5d["coverage"] == 1.0
    assert energy_5d["basket_return"] == pytest.approx((1.004**5) - 1.0)
    assert energy_5d["qqq_return"] == pytest.approx((1.001**5) - 1.0)
    assert energy_5d["breadth"] == 1.0
    assert energy_5d["annualized_volatility"] == pytest.approx(0.0)
    assert energy_5d["max_drawdown"] == pytest.approx(0.0)

    summary = diagnostics.summary.iloc[0]
    assert summary["market_state"] == "Rotation"
    assert summary["classification"] == "healthy rotation"
    assert summary["rotation_direction"] == "Applications → Energy"
    assert summary["positive_layer_count"] == 3
    assert summary["negative_layer_count"] == 2


def test_one_positive_layer_is_crowded_rotation() -> None:
    summary = summarize_ai_layer_rotation(_summary_detail([0.03, -0.01, -0.02, -0.03, -0.04])).iloc[0]

    assert summary["market_state"] == "Rotation"
    assert summary["classification"] == "crowded rotation"
    assert summary["positive_layer_count"] == 1
    assert summary["all_layers_weaker"] == False


def test_five_negative_layers_are_ai_risk_off() -> None:
    summary = summarize_ai_layer_rotation(_summary_detail([-0.01, -0.02, -0.03, -0.04, -0.05])).iloc[0]

    assert summary["market_state"] == "AI risk-off"
    assert summary["classification"] == "broad de-risking"
    assert summary["rotation_direction"] == "All layers lower"
    assert summary["all_layers_weaker"] == True


def test_missing_layer_data_stays_unavailable() -> None:
    diagnostics = build_ai_layer_rotation_diagnostics({"QQQ": _price_frame(0.001)})

    assert len(diagnostics.detail) == 20
    assert diagnostics.detail["basket_return"].isna().all()
    assert diagnostics.summary.loc[0, "market_state"] == "Unavailable"
    assert diagnostics.summary.loc[0, "classification"] == "insufficient_data"


def test_display_formatting_and_export_keep_numeric_values(tmp_path) -> None:
    diagnostics = build_ai_layer_rotation_diagnostics(_rotation_frames())
    formatted = format_ai_layer_rotation_display(diagnostics.detail)

    assert formatted.loc[0, "basket_return"].endswith("%")
    assert isinstance(diagnostics.detail.loc[0, "basket_return"], float)

    result = export_research_lab_diagnostics(
        run_metadata={"benchmark": "QQQ"},
        tables={
            "ai_layer_rotation_diagnostics": diagnostics.detail,
            "ai_layer_rotation_summary": diagnostics.summary,
        },
        output_root=tmp_path,
        run_id="run",
    )
    exported = pd.read_csv(result.run_dir / "ai_layer_rotation_diagnostics.csv")
    assert pd.api.types.is_numeric_dtype(exported["relative_qqq_return"])
    assert result.manifest["row_counts"]["ai_layer_rotation_diagnostics.csv"] == 20
    assert result.manifest["row_counts"]["ai_layer_rotation_summary.csv"] == 1
