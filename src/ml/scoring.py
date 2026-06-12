"""Current-row ML scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.datasets import assert_no_label_leakage, latest_feature_rows
from src.ml.models import fit_classifier, predict_positive_probability


def ml_score(outperform_probability: pd.Series, drawdown_risk_probability: pd.Series) -> pd.Series:
    """Convert opportunity and risk probabilities into a 0-100 advisory score."""

    outperformance_evidence = np.sqrt((1.0 - outperform_probability).clip(lower=0.0, upper=1.0))
    score = 100.0 * (0.7 * outperformance_evidence + 0.3 * (1.0 - drawdown_risk_probability))
    return score.clip(lower=0, upper=100)


def confidence_bucket(outperform_probability: pd.Series, drawdown_risk_probability: pd.Series) -> pd.Series:
    """Bucket confidence by distance from an undecided 50% probability."""

    distance = pd.concat(
        [(outperform_probability - 0.5).abs(), (drawdown_risk_probability - 0.5).abs()],
        axis=1,
    ).max(axis=1)
    return pd.cut(
        distance,
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)


def current_ml_score_table(
    dataset: pd.DataFrame,
    feature_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    model_name: str,
    horizon: int = 20,
) -> pd.DataFrame:
    """Train advisory label models on available history and score latest rows."""

    assert_no_label_leakage(feature_columns)
    latest = latest_feature_rows(feature_frames)
    if dataset.empty or latest.empty or not feature_columns:
        return pd.DataFrame()

    out_label = f"label_outperform_{horizon}d"
    risk_label = f"label_drawdown_risk_{horizon}d"
    out_model = fit_classifier(dataset, feature_columns, out_label, model_name)
    risk_model = fit_classifier(dataset, feature_columns, risk_label, model_name)

    out_prob = pd.Series(predict_positive_probability(out_model, latest[feature_columns]), index=latest.index)
    risk_prob = pd.Series(predict_positive_probability(risk_model, latest[feature_columns]), index=latest.index)
    output = pd.DataFrame(
        {
            "Ticker": latest["Ticker"],
            "Date": latest["Date"],
            "Rule-Based Regime": latest.get("regime"),
            "ML Outperformance Probability": out_prob,
            "ML Drawdown-Risk Probability": risk_prob,
            "ML Score": ml_score(out_prob, risk_prob),
            "Confidence": confidence_bucket(out_prob, risk_prob),
            "Risk Flags": latest.get("risk_flags", ""),
        }
    )
    return output.sort_values("ML Score", ascending=False).reset_index(drop=True)
