from __future__ import annotations

import pandas as pd

from src.ml.interpretations import (
    interpret_drawdown_calibration,
    interpret_ml_diagnostics_summary,
    interpret_ml_score_buckets,
)


def score_buckets(
    low_hit: float = 0.30,
    high_hit: float = 0.70,
    low_return: float = -0.02,
    high_return: float = 0.05,
    count: int = 6,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "score_bucket": "Low",
                "count": count,
                "outperformance_hit_rate": low_hit,
                "average_forward_return": low_return,
            },
            {
                "score_bucket": "High",
                "count": count,
                "outperformance_hit_rate": high_hit,
                "average_forward_return": high_return,
            },
        ]
    )


def drawdown_calibration(
    low_rate: float = 0.10,
    high_rate: float = 0.60,
    count: int = 6,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "average_probability": 0.20,
                "observed_drawdown_risk_rate": low_rate,
                "count": count,
            },
            {
                "average_probability": 0.80,
                "observed_drawdown_risk_rate": high_rate,
                "count": count,
            },
        ]
    )


def test_interpret_ml_score_buckets_identifies_good_separation() -> None:
    interpretation = interpret_ml_score_buckets(score_buckets())

    assert interpretation.label == "Good separation"
    assert interpretation.level == "success"
    assert "secondary research input" in interpretation.message
    assert "buy" not in interpretation.message.lower()


def test_interpret_ml_score_buckets_warns_when_high_bucket_is_worse() -> None:
    interpretation = interpret_ml_score_buckets(
        score_buckets(low_hit=0.70, high_hit=0.30, low_return=0.05, high_return=-0.02)
    )

    assert interpretation.label == "Treat ML score cautiously"
    assert interpretation.level == "warning"
    assert "sell" not in interpretation.message.lower()


def test_interpret_ml_score_buckets_handles_small_samples() -> None:
    interpretation = interpret_ml_score_buckets(score_buckets(count=2))

    assert interpretation.label == "Insufficient sample"
    assert "more observations" in interpretation.message


def test_interpret_drawdown_calibration_identifies_useful_separation() -> None:
    interpretation = interpret_drawdown_calibration(drawdown_calibration())

    assert interpretation.label == "Risk calibration looks useful"
    assert interpretation.level == "success"
    assert "useful separation" in interpretation.message


def test_interpret_drawdown_calibration_warns_on_tail_risk() -> None:
    interpretation = interpret_drawdown_calibration(drawdown_calibration(low_rate=0.60, high_rate=0.75))

    assert interpretation.label == "Tail risk may be understated"
    assert interpretation.level == "warning"
    assert "cautious evidence" in interpretation.message


def test_interpret_drawdown_calibration_handles_small_samples() -> None:
    interpretation = interpret_drawdown_calibration(drawdown_calibration(count=2))

    assert interpretation.label == "Insufficient sample"


def test_interpret_ml_diagnostics_summary_explains_research_only_coverage() -> None:
    summary = pd.DataFrame(
        [
            {"target": "outperformance", "predictions": 30, "folds": 3},
            {"target": "drawdown_risk", "predictions": 30, "folds": 3},
        ]
    )

    interpretation = interpret_ml_diagnostics_summary(summary)

    assert interpretation.label == "Research coverage available"
    assert "does not create buy or sell instructions" in interpretation.message
