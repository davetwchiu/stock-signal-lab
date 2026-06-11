"""Diagnostics for existing out-of-sample ML signal outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml.metrics import calibration_table
from src.ml.scoring import ml_score


BASELINE_COMPARISON_COLUMNS = [
    "signal",
    "sample_size",
    "top_bucket_forward_return",
    "bottom_bucket_forward_return",
    "spread",
    "direction",
    "interpretation",
]

SIMPLE_BASELINE_CANDIDATES = (
    ("return_60d", "Momentum (60d)"),
    ("return_20d", "Momentum (20d)"),
    ("rs_spy_60d", "Relative strength (SPY 60d)"),
    ("rs_qqq_60d", "Relative strength (QQQ 60d)"),
    ("dist_ma_200d", "Trend (distance from 200d MA)"),
    ("dist_ma_50d", "Trend (distance from 50d MA)"),
    ("rsi_14d", "Momentum (RSI 14d)"),
)

MIN_BASELINE_BUCKET_COUNT = 5
SPREAD_SIMILARITY_TOLERANCE = 0.0025


@dataclass(frozen=True)
class MLDiagnostics:
    """Diagnostic tables for the existing advisory ML signal."""

    score_buckets: pd.DataFrame
    drawdown_risk_calibration: pd.DataFrame
    summary: pd.DataFrame
    baseline_comparison: pd.DataFrame


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


def _empty_baseline_comparison() -> pd.DataFrame:
    return pd.DataFrame(columns=BASELINE_COMPARISON_COLUMNS)


def _comparison_direction(spread: object) -> str:
    if pd.isna(spread):
        return "insufficient"
    if spread > 0:
        return "top bucket higher"
    if spread < 0:
        return "bottom bucket higher"
    return "flat"


def _comparison_row(
    signal: str,
    sample_size: int,
    top_return: float | None,
    bottom_return: float | None,
) -> dict[str, object]:
    spread = pd.NA
    if top_return is not None and bottom_return is not None:
        spread = float(top_return - bottom_return)
    return {
        "signal": signal,
        "sample_size": int(sample_size),
        "top_bucket_forward_return": pd.NA if top_return is None else float(top_return),
        "bottom_bucket_forward_return": pd.NA if bottom_return is None else float(bottom_return),
        "spread": spread,
        "direction": _comparison_direction(spread),
        "interpretation": "The sample is too small for a reliable baseline comparison."
        if pd.isna(spread)
        else "",
    }


def _preferred_simple_baseline(data: pd.DataFrame) -> tuple[str, str] | None:
    for column, label in SIMPLE_BASELINE_CANDIDATES:
        if column in data and data[column].notna().sum() >= MIN_BASELINE_BUCKET_COUNT * 2:
            return column, label
    return None


def _merge_baseline_panel(score_panel: pd.DataFrame, baseline_panel: pd.DataFrame | None) -> pd.DataFrame:
    if baseline_panel is None or baseline_panel.empty:
        return score_panel.copy()
    if "Date" not in score_panel or "Ticker" not in score_panel:
        return score_panel.copy()
    if "Date" not in baseline_panel or "Ticker" not in baseline_panel:
        return score_panel.copy()

    candidate_columns = [column for column, _ in SIMPLE_BASELINE_CANDIDATES if column in baseline_panel]
    if not candidate_columns:
        return score_panel.copy()

    left = score_panel.copy()
    right = baseline_panel[["Date", "Ticker", *candidate_columns]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])
    right = right.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return left.merge(right, on=["Date", "Ticker"], how="left")


def _ml_score_comparison_row(
    data: pd.DataFrame,
    forward_return_column: str,
    min_bucket_size: int,
) -> dict[str, object]:
    if "ML Score" not in data or forward_return_column not in data:
        return _comparison_row("ML score", 0, None, None)

    usable = data.dropna(subset=["ML Score", forward_return_column]).copy()
    if usable.empty or len(usable) < min_bucket_size * 2:
        return _comparison_row("ML score", len(usable), None, None)

    usable["score_bucket"] = pd.cut(
        usable["ML Score"],
        bins=[-float("inf"), 40.0, 70.0, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    low = usable[usable["score_bucket"] == "Low"]
    high = usable[usable["score_bucket"] == "High"]
    if len(low) < min_bucket_size or len(high) < min_bucket_size:
        return _comparison_row("ML score", len(usable), None, None)
    return _comparison_row(
        "ML score",
        len(usable),
        high[forward_return_column].mean(),
        low[forward_return_column].mean(),
    )


def _no_skill_comparison_row(
    data: pd.DataFrame,
    forward_return_column: str,
    min_bucket_size: int,
) -> dict[str, object]:
    if forward_return_column not in data:
        return _comparison_row("No-skill / universe average", 0, None, None)
    usable = data.dropna(subset=[forward_return_column])
    if len(usable) < min_bucket_size * 2:
        return _comparison_row("No-skill / universe average", len(usable), None, None)
    average_return = float(usable[forward_return_column].mean())
    return _comparison_row(
        "No-skill / universe average",
        len(usable),
        average_return,
        average_return,
    )


def _simple_baseline_comparison_row(
    data: pd.DataFrame,
    baseline_column: str,
    signal: str,
    forward_return_column: str,
    min_bucket_size: int,
) -> dict[str, object]:
    if baseline_column not in data or forward_return_column not in data:
        return _comparison_row(signal, 0, None, None)

    usable = data.dropna(subset=[baseline_column, forward_return_column]).copy()
    if len(usable) < min_bucket_size * 2 or usable[baseline_column].nunique() < 2:
        return _comparison_row(signal, len(usable), None, None)

    try:
        usable["baseline_bucket"] = pd.qcut(
            usable[baseline_column],
            q=3,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return _comparison_row(signal, len(usable), None, None)

    top_bucket = usable["baseline_bucket"].max()
    bottom_bucket = usable["baseline_bucket"].min()
    if pd.isna(top_bucket) or pd.isna(bottom_bucket) or top_bucket == bottom_bucket:
        return _comparison_row(signal, len(usable), None, None)

    bottom = usable[usable["baseline_bucket"] == bottom_bucket]
    top = usable[usable["baseline_bucket"] == top_bucket]
    if len(bottom) < min_bucket_size or len(top) < min_bucket_size:
        return _comparison_row(signal, len(usable), None, None)
    return _comparison_row(
        signal,
        len(usable),
        top[forward_return_column].mean(),
        bottom[forward_return_column].mean(),
    )


def _spread_value(row: pd.Series) -> float | None:
    value = pd.to_numeric(pd.Series([row.get("spread")]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def _add_baseline_interpretations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows

    ml_rows = frame[frame["signal"] == "ML score"]
    ml_spread = None if ml_rows.empty else _spread_value(ml_rows.iloc[0])
    baseline_spreads = [
        _spread_value(row)
        for _, row in frame[frame["signal"] != "ML score"].iterrows()
    ]
    usable_baseline_spreads = [value for value in baseline_spreads if value is not None]
    best_baseline_spread = max(usable_baseline_spreads) if usable_baseline_spreads else 0.0

    interpreted: list[dict[str, object]] = []
    for row in rows:
        updated = row.copy()
        spread = pd.to_numeric(pd.Series([updated.get("spread")]), errors="coerce").iloc[0]
        if pd.isna(spread):
            updated["interpretation"] = "The sample is too small for a reliable baseline comparison."
        elif updated["signal"] == "No-skill / universe average":
            updated["interpretation"] = (
                "No-skill reference uses the same average forward return for top and bottom buckets."
            )
        elif updated["signal"] == "ML score":
            if ml_spread is None:
                updated["interpretation"] = "The sample is too small for a reliable baseline comparison."
            elif ml_spread > best_baseline_spread + SPREAD_SIMILARITY_TOLERANCE:
                updated["interpretation"] = (
                    "ML score shows better bucket spread than the simple baseline in this sample."
                )
            elif ml_spread + SPREAD_SIMILARITY_TOLERANCE < best_baseline_spread:
                updated["interpretation"] = (
                    "ML score does not improve on the simple baseline. Treat ML evidence cautiously."
                )
            else:
                updated["interpretation"] = (
                    "The baseline and ML score are similar. This suggests the ML score may mostly be "
                    "capturing simple trend or momentum."
                )
        elif ml_spread is None:
            updated["interpretation"] = "The sample is too small for a reliable baseline comparison."
        elif float(spread) > ml_spread + SPREAD_SIMILARITY_TOLERANCE:
            updated["interpretation"] = (
                "This simple baseline shows better bucket spread than ML score in this sample."
            )
        elif float(spread) + SPREAD_SIMILARITY_TOLERANCE < ml_spread:
            updated["interpretation"] = (
                "ML score shows better bucket spread than this simple baseline in this sample."
            )
        else:
            updated["interpretation"] = (
                "The baseline and ML score are similar. This suggests the ML score may mostly be "
                "capturing simple trend or momentum."
            )
        interpreted.append(updated)
    return interpreted


def build_ml_baseline_comparison(
    score_panel: pd.DataFrame,
    baseline_panel: pd.DataFrame | None = None,
    forward_return_column: str = "forward_return",
    min_bucket_size: int = MIN_BASELINE_BUCKET_COUNT,
) -> pd.DataFrame:
    """Return a compact DataFrame comparing ML score bucket separation with simple baselines."""

    if score_panel.empty:
        return _empty_baseline_comparison()

    data = _merge_baseline_panel(score_panel, baseline_panel)
    rows = [
        _ml_score_comparison_row(data, forward_return_column, min_bucket_size),
        _no_skill_comparison_row(data, forward_return_column, min_bucket_size),
    ]
    simple_baseline = _preferred_simple_baseline(data)
    if simple_baseline is not None:
        baseline_column, signal = simple_baseline
        rows.append(
            _simple_baseline_comparison_row(
                data,
                baseline_column,
                signal,
                forward_return_column,
                min_bucket_size,
            )
        )

    return pd.DataFrame(_add_baseline_interpretations(rows), columns=BASELINE_COMPARISON_COLUMNS)


def build_ml_diagnostics(
    outperformance_predictions: pd.DataFrame,
    drawdown_risk_predictions: pd.DataFrame,
    outperformance_metrics: pd.DataFrame | None = None,
    drawdown_risk_metrics: pd.DataFrame | None = None,
    risk_bins: int = 5,
    baseline_panel: pd.DataFrame | None = None,
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
        return MLDiagnostics(pd.DataFrame(), pd.DataFrame(), summary, _empty_baseline_comparison())

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
    baseline_comparison = build_ml_baseline_comparison(score_panel, baseline_panel)

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
    return MLDiagnostics(_score_bucket_summary(score_panel), risk_calibration, summary, baseline_comparison)
