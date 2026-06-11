"""Stability scoring for robustness experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stability_score(values: pd.Series) -> float:
    """Return a 0-100 stability score based on dispersion and sign consistency."""

    clean = values.dropna().astype(float)
    if clean.empty:
        return np.nan
    dispersion = float(clean.std())
    sign_consistency = max(float((clean >= 0).mean()), float((clean < 0).mean()))
    raw = 100.0 * sign_consistency / (1.0 + 10.0 * dispersion)
    return float(max(0.0, min(100.0, raw)))


def warning_flags(results: pd.DataFrame, metric: str = "Sharpe") -> list[str]:
    """Generate plain-English warning flags for robustness results."""

    if results.empty or metric not in results:
        return ["no robustness results"]
    values = results[metric].dropna().astype(float)
    flags: list[str] = []
    if values.empty:
        return ["selected metric unavailable"]
    if values.std() > max(abs(values.median()), 0.1):
        flags.append("high performance dispersion")
    if (values > 0).mean() < 0.60:
        flags.append("weak sign consistency")
    if values.max() > values.median() * 2 and values.median() > 0:
        flags.append("best case much stronger than median")
    return flags or ["no narrow-parameter warning"]


def summarize_robustness(results: pd.DataFrame, metric: str = "Sharpe") -> pd.DataFrame:
    """Return best, median, worst, dispersion, and warning fields."""

    if results.empty or metric not in results:
        return pd.DataFrame()
    values = results[metric].dropna().astype(float)
    if values.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "test_cases": len(values),
                "best_result": values.max(),
                "median_result": values.median(),
                "worst_result": values.min(),
                "dispersion": values.std(),
                "stability_score": stability_score(values),
                "warning_flags": ", ".join(warning_flags(results, metric)),
            }
        ]
    )

