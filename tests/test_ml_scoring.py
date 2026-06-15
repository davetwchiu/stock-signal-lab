from __future__ import annotations

import pandas as pd
import pytest

from src.ml.scoring import current_ml_score_table, ml_score


def test_ml_score_uses_corrected_outperformance_probability_direction() -> None:
    outperform_probability = pd.Series([0.20, 0.80], index=["weak", "strong"])
    drawdown_risk_probability = pd.Series([0.30, 0.30], index=outperform_probability.index)

    scores = ml_score(outperform_probability, drawdown_risk_probability)

    assert scores.loc["strong"] > scores.loc["weak"]


def test_ml_score_equals_raw_opportunity_probability_scaled_to_100() -> None:
    outperform_probability = pd.Series([0.10, 0.55, 0.90], index=["low", "middle", "high"])
    drawdown_risk_probability = pd.Series([0.80, 0.20, 0.60], index=outperform_probability.index)

    scores = ml_score(outperform_probability, drawdown_risk_probability)

    pd.testing.assert_series_equal(scores, pd.Series([10.0, 55.0, 90.0], index=outperform_probability.index))


def test_ml_score_does_not_directly_change_for_drawdown_risk_probability() -> None:
    outperform_probability = pd.Series([0.40, 0.40], index=["lower_risk", "higher_risk"])
    drawdown_risk_probability = pd.Series([0.20, 0.80], index=outperform_probability.index)

    scores = ml_score(outperform_probability, drawdown_risk_probability)

    assert scores.loc["lower_risk"] == scores.loc["higher_risk"] == 40.0


def test_ml_score_remains_clipped_to_score_range() -> None:
    outperform_probability = pd.Series([-0.50, 1.50], index=["below", "above"])
    drawdown_risk_probability = pd.Series([1.50, -0.50], index=outperform_probability.index)

    scores = ml_score(outperform_probability, drawdown_risk_probability)

    assert scores.loc["below"] == 0.0
    assert scores.loc["above"] == 100.0


def test_current_ml_score_table_keeps_drawdown_risk_probability_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0],
            "label_outperform_20d": [0, 1, 0, 1],
            "label_drawdown_risk_20d": [1, 0, 1, 0],
        }
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    feature_frames = {
        "LOW": pd.DataFrame({"feature": [1.0], "regime": ["Weak"], "risk_flags": ["Elevated"]}, index=[dates[0]]),
        "HIGH": pd.DataFrame({"feature": [2.0], "regime": ["Strong"], "risk_flags": ["Clear"]}, index=[dates[1]]),
    }

    def fake_fit_classifier(
        data: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        model_name: str,
    ) -> str:
        return label_column

    def fake_predict_positive_probability(model: str, features: pd.DataFrame) -> list[float]:
        if model == "label_outperform_20d":
            return [0.25, 0.75]
        return [0.90, 0.10]

    monkeypatch.setattr("src.ml.scoring.fit_classifier", fake_fit_classifier)
    monkeypatch.setattr("src.ml.scoring.predict_positive_probability", fake_predict_positive_probability)

    table = current_ml_score_table(dataset, feature_frames, ["feature"], "logistic_regression")

    indexed = table.set_index("Ticker")
    assert indexed.loc["LOW", "ML Score"] == 25.0
    assert indexed.loc["HIGH", "ML Score"] == 75.0
    assert indexed.loc["LOW", "ML Drawdown-Risk Probability"] == 0.90
    assert indexed.loc["HIGH", "ML Drawdown-Risk Probability"] == 0.10
