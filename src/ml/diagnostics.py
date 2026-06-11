"""Diagnostics for existing out-of-sample ML signal outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml.metrics import calibration_table
from src.ml.scoring import ml_score


@dataclass(frozen=True)
class MLDiagnostics:
    """Diagnostic tables for the existing advisory ML signal."""

    score_buckets: pd.DataFrame
    drawdown_risk_calibration: pd.DataFrame
    summary: pd.DataFrame


def _overall_summary(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame | None,
    target: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "target": target,
        "predictions": len(predictions),
        "folds": int(predictions["fold"].nunique()) if "fold" in predictions else 0,
        "positive_rate": (
            predictions["actual"].mean() if "actual" in predictions and not predictions.empty else pd.NA
        ),
        "start": predictions["Date"].min() if "Date" in predictions and not predictions.empty else pd.NaT,
        "end": predictions["Date"].max() if "Date" in predictions and not predictions.empty else pd.NaT,
    }
    if metrics is not None and not metrics.empty:
        row.update(metrics.iloc[0].to_dict())
    return row


def _score_bucket_summary(score_panel: pd.DataFrame) -> pd.DataFrame:
    if score_panel.empty or score_panel["ML Score"].nunique() < 2:
        return pd.DataFrame()

    data = score_panel.dropna(subset=["ML Score", "actual_out"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["score_bucket"] = pd.cut(
        data["ML Score"],
        bins=[-float("inf"), 40.0, 70.0, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    aggregations = {
        "count": ("actual_out", "size"),
        "average_ml_score": ("ML Score", "mean"),
        "outperformance_hit_rate": ("actual_out", "mean"),
        "average_outperformance_probability": ("probability_out", "mean"),
        "average_drawdown_risk_probability": ("probability_risk", "mean"),
    }
    optional_columns = {
        "actual_risk": ("drawdown_risk_rate", "mean"),
        "forward_return": ("average_forward_return", "mean"),
        "forward_excess_return": ("average_forward_excess_return", "mean"),
        "forward_drawdown": ("average_forward_drawdown", "mean"),
    }
    for column, (output_name, operation) in optional_columns.items():
        if column in data:
            aggregations[output_name] = (column, operation)

    return data.groupby("score_bucket", observed=True).agg(**aggregations).reset_index()


def build_ml_diagnostics(
    outperformance_predictions: pd.DataFrame,
    drawdown_risk_predictions: pd.DataFrame,
    outperformance_metrics: pd.DataFrame | None = None,
    drawdown_risk_metrics: pd.DataFrame | None = None,
    risk_bins: int = 5,
) -> MLDiagnostics:
    """Summarize out-of-sample usefulness of the existing ML signal.

    Inputs should come from existing walk-forward validation results. This helper
    only joins predictions and summarizes diagnostics; it does not fit models or
    change score, probability, or decision logic.
    """

    if outperformance_predictions.empty or drawdown_risk_predictions.empty:
        summary = pd.DataFrame(
            [
                _overall_summary(outperformance_predictions, outperformance_metrics, "outperformance"),
                _overall_summary(drawdown_risk_predictions, drawdown_risk_metrics, "drawdown_risk"),
            ]
        )
        return MLDiagnostics(pd.DataFrame(), pd.DataFrame(), summary)

    out_columns = [
        "Date",
        "Ticker",
        "actual",
        "probability",
        "forward_return",
        "forward_excess_return",
        "forward_drawdown",
    ]
    risk_columns = ["Date", "Ticker", "actual", "probability"]
    out = outperformance_predictions[
        [column for column in out_columns if column in outperformance_predictions]
    ].rename(columns={"actual": "actual_out", "probability": "probability_out"})
    risk = drawdown_risk_predictions[
        [column for column in risk_columns if column in drawdown_risk_predictions]
    ].rename(
        columns={"actual": "actual_risk", "probability": "probability_risk"}
    )
    score_panel = out.merge(risk, on=["Date", "Ticker"], how="inner")
    if not score_panel.empty:
        score_panel["ML Score"] = ml_score(score_panel["probability_out"], score_panel["probability_risk"])

    risk_calibration = calibration_table(
        drawdown_risk_predictions,
        probability_column="probability",
        label_column="actual",
        bins=risk_bins,
    ).rename(columns={"observed_rate": "observed_drawdown_risk_rate"})
    summary = pd.DataFrame(
        [
            _overall_summary(outperformance_predictions, outperformance_metrics, "outperformance"),
            _overall_summary(drawdown_risk_predictions, drawdown_risk_metrics, "drawdown_risk"),
        ]
    )
    return MLDiagnostics(_score_bucket_summary(score_panel), risk_calibration, summary)
