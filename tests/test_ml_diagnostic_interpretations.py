from __future__ import annotations

import pandas as pd

from src.ml.interpretations import (
    interpret_drawdown_calibration,
    interpret_ml_diagnostics_summary,
    interpret_research_lab_run,
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


def diagnostics_summary(
    predictions: int = 40,
    folds: int = 3,
    roc_auc: float = 0.62,
    f1: float = 0.50,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "outperformance",
                "predictions": predictions,
                "folds": folds,
                "roc_auc": roc_auc,
                "f1": f1,
            },
            {
                "target": "drawdown_risk",
                "predictions": predictions,
                "folds": folds,
                "roc_auc": roc_auc,
                "f1": f1,
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


def test_interpret_research_lab_run_identifies_useful_evidence() -> None:
    interpretation = interpret_research_lab_run(
        diagnostics_summary(),
        score_buckets(),
        drawdown_calibration(),
    )

    assert interpretation.overall == "Usable but not strong research evidence."
    assert interpretation.level == "success"
    assert "usable directional evidence" in interpretation.walk_forward_validation
    assert "useful ranking evidence" in interpretation.ml_score_buckets
    assert "useful as a caution flag" in interpretation.drawdown_risk_calibration
    assert "secondary research evidence" in interpretation.use


def test_interpret_research_lab_run_identifies_mixed_evidence() -> None:
    interpretation = interpret_research_lab_run(
        diagnostics_summary(roc_auc=0.54, f1=0.40),
        score_buckets(low_hit=0.40, high_hit=0.55, low_return=0.03, high_return=0.02),
        drawdown_calibration(),
    )

    assert interpretation.overall == "Mixed research evidence."
    assert interpretation.level == "info"
    assert "mixed" in interpretation.walk_forward_validation.lower()
    assert "some outcomes but not others" in interpretation.ml_score_buckets


def test_interpret_research_lab_run_identifies_weak_bucket_separation() -> None:
    interpretation = interpret_research_lab_run(
        diagnostics_summary(),
        score_buckets(low_hit=0.50, high_hit=0.51, low_return=0.01, high_return=0.012),
        drawdown_calibration(),
    )

    assert interpretation.overall == "Mixed research evidence."
    assert "did not clearly outperform" in interpretation.ml_score_buckets


def test_interpret_research_lab_run_identifies_weak_drawdown_calibration() -> None:
    interpretation = interpret_research_lab_run(
        diagnostics_summary(),
        score_buckets(),
        drawdown_calibration(low_rate=0.40, high_rate=0.39),
    )

    assert interpretation.overall == "Mixed research evidence."
    assert "do not clearly separate" in interpretation.drawdown_risk_calibration


def test_interpret_research_lab_run_handles_insufficient_data() -> None:
    interpretation = interpret_research_lab_run(
        diagnostics_summary(predictions=4, folds=1),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert interpretation.overall == "Inconclusive research evidence."
    assert interpretation.level == "warning"
    assert "too small" in interpretation.walk_forward_validation
    assert "too little usable data" in interpretation.ml_score_buckets
    assert "too small or incomplete" in interpretation.drawdown_risk_calibration


def test_interpret_research_lab_run_avoids_trading_wording() -> None:
    interpretation = interpret_research_lab_run(
        diagnostics_summary(),
        score_buckets(),
        drawdown_calibration(),
    )
    text = " ".join(
        [
            interpretation.overall,
            interpretation.walk_forward_validation,
            interpretation.ml_score_buckets,
            interpretation.drawdown_risk_calibration,
            interpretation.use,
        ]
    ).lower()

    for forbidden in ("buy", "sell", "add", "reduce", "increase position size", "change allocation"):
        assert forbidden not in text
