from __future__ import annotations

import pandas as pd

from src.ml.scoring import ml_score


def test_ml_score_uses_corrected_outperformance_probability_direction() -> None:
    outperform_probability = pd.Series([0.80, 0.20], index=["inverted_weak", "inverted_strong"])
    drawdown_risk_probability = pd.Series([0.30, 0.30], index=outperform_probability.index)

    scores = ml_score(outperform_probability, drawdown_risk_probability)

    assert scores.loc["inverted_strong"] > scores.loc["inverted_weak"]


def test_ml_score_keeps_drawdown_risk_probability_penalty_direction() -> None:
    outperform_probability = pd.Series([0.40, 0.40], index=["lower_risk", "higher_risk"])
    drawdown_risk_probability = pd.Series([0.20, 0.80], index=outperform_probability.index)

    scores = ml_score(outperform_probability, drawdown_risk_probability)

    assert scores.loc["lower_risk"] > scores.loc["higher_risk"]
