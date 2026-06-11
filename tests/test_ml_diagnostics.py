from __future__ import annotations

import pandas as pd

from src.ml.diagnostics import build_ml_diagnostics


def validation_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    outperformance = pd.DataFrame(
        {
            "fold": [1, 1, 1, 2, 2, 2],
            "Date": dates,
            "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "actual": [0, 0, 0, 1, 1, 1],
            "probability": [0.20, 0.30, 0.60, 0.65, 0.85, 0.90],
            "prediction": [0, 0, 1, 1, 1, 1],
            "forward_return": [-0.05, -0.02, 0.01, 0.03, 0.08, 0.10],
            "forward_excess_return": [-0.08, -0.04, -0.01, 0.02, 0.06, 0.09],
            "forward_drawdown": [-0.20, -0.15, -0.08, -0.07, -0.04, -0.03],
        }
    )
    drawdown_risk = pd.DataFrame(
        {
            "fold": [1, 1, 1, 2, 2, 2],
            "Date": dates,
            "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "actual": [1, 1, 0, 0, 0, 0],
            "probability": [0.80, 0.75, 0.45, 0.35, 0.20, 0.10],
            "prediction": [1, 1, 0, 0, 0, 0],
        }
    )
    return outperformance, drawdown_risk


def test_ml_diagnostics_bucket_existing_out_of_sample_scores() -> None:
    outperformance, drawdown_risk = validation_predictions()

    diagnostics = build_ml_diagnostics(outperformance, drawdown_risk)
    buckets = diagnostics.score_buckets.set_index("score_bucket")

    assert buckets.loc["Low", "count"] == 2
    assert buckets.loc["Medium", "count"] == 2
    assert buckets.loc["High", "count"] == 2
    assert buckets.loc["Low", "outperformance_hit_rate"] == 0.0
    assert buckets.loc["High", "outperformance_hit_rate"] == 1.0
    assert (
        buckets.loc["High", "average_forward_excess_return"]
        > buckets.loc["Low", "average_forward_excess_return"]
    )
    assert buckets.loc["Low", "average_drawdown_risk_probability"] > buckets.loc[
        "High", "average_drawdown_risk_probability"
    ]


def test_ml_diagnostics_include_drawdown_risk_calibration_and_summary() -> None:
    outperformance, drawdown_risk = validation_predictions()
    out_metrics = pd.DataFrame([{"accuracy": 0.5, "roc_auc": 0.7}])
    risk_metrics = pd.DataFrame([{"accuracy": 0.8, "roc_auc": 0.75}])

    diagnostics = build_ml_diagnostics(outperformance, drawdown_risk, out_metrics, risk_metrics, risk_bins=2)

    assert list(diagnostics.summary["target"]) == ["outperformance", "drawdown_risk"]
    assert diagnostics.summary.loc[0, "predictions"] == 6
    assert diagnostics.summary.loc[0, "folds"] == 2
    assert diagnostics.summary.loc[1, "accuracy"] == 0.8
    assert "observed_drawdown_risk_rate" in diagnostics.drawdown_risk_calibration.columns
    assert diagnostics.drawdown_risk_calibration["count"].sum() == 6


def test_ml_diagnostics_handle_empty_predictions() -> None:
    diagnostics = build_ml_diagnostics(pd.DataFrame(), pd.DataFrame())

    assert diagnostics.score_buckets.empty
    assert diagnostics.drawdown_risk_calibration.empty
    assert list(diagnostics.summary["target"]) == ["outperformance", "drawdown_risk"]
    assert diagnostics.summary["predictions"].tolist() == [0, 0]
