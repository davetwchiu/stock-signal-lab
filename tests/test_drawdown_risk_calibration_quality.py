from __future__ import annotations

import re

import pandas as pd
import pytest

from src.ml.diagnostics import build_drawdown_risk_calibration_quality


def calibration(
    *,
    count: int = 10,
    predicted: tuple[float, float, float] = (0.10, 0.50, 0.80),
    observed: tuple[float, float, float] = (0.10, 0.50, 0.80),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "count": [count, count, count],
            "average_probability": list(predicted),
            "observed_drawdown_risk_rate": list(observed),
        }
    )


def row_level_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "probability": [0.10, 0.20, 0.70, 0.80],
            "actual": [0, 0, 1, 1],
        }
    )


def quality_row(quality: pd.DataFrame) -> pd.Series:
    assert len(quality) == 1
    return quality.iloc[0]


def test_drawdown_risk_calibration_quality_identifies_aligned_sample() -> None:
    quality = build_drawdown_risk_calibration_quality(calibration(), row_level_predictions())

    row = quality_row(quality)

    assert row["sample_size"] == 30
    assert row["mean_predicted_risk"] == pytest.approx(0.4666666667)
    assert row["observed_drawdown_rate"] == pytest.approx(0.4666666667)
    assert row["calibration_gap"] == pytest.approx(0.0)
    assert row["mean_absolute_calibration_error"] == pytest.approx(0.0)
    assert row["max_bucket_calibration_gap"] == pytest.approx(0.0)
    assert row["brier_score"] == pytest.approx(0.045)
    assert row["monotonicity"] == "higher buckets aligned"
    assert row["interpretation"] == "Drawdown-risk calibration looks broadly aligned in this sample."


def test_drawdown_risk_calibration_quality_flags_underestimated_risk() -> None:
    quality = build_drawdown_risk_calibration_quality(
        calibration(
            predicted=(0.20, 0.40, 0.60),
            observed=(0.40, 0.60, 0.80),
        )
    )

    row = quality_row(quality)

    assert row["calibration_gap"] == pytest.approx(0.20)
    assert row["interpretation"] == "The model appears to underestimate realised drawdown risk in this sample."


def test_drawdown_risk_calibration_quality_flags_overestimated_risk() -> None:
    quality = build_drawdown_risk_calibration_quality(
        calibration(
            predicted=(0.40, 0.60, 0.80),
            observed=(0.20, 0.40, 0.60),
        )
    )

    row = quality_row(quality)

    assert row["calibration_gap"] == pytest.approx(-0.20)
    assert row["interpretation"] == "The model appears to overestimate realised drawdown risk in this sample."


def test_drawdown_risk_calibration_quality_flags_poor_monotonicity() -> None:
    quality = build_drawdown_risk_calibration_quality(
        calibration(
            predicted=(0.20, 0.50, 0.80),
            observed=(0.20, 0.70, 0.30),
        )
    )

    row = quality_row(quality)

    assert row["monotonicity"] == "not clearly monotonic"
    assert row["interpretation"] == (
        "Higher predicted-risk buckets do not show clearly higher realised drawdown rates."
    )


def test_drawdown_risk_calibration_quality_handles_insufficient_sample() -> None:
    quality = build_drawdown_risk_calibration_quality(calibration(count=2))

    row = quality_row(quality)

    assert row["sample_size"] == 6
    assert row["interpretation"] == "The sample is too small for reliable calibration quality assessment."


def test_drawdown_risk_calibration_quality_handles_missing_required_columns() -> None:
    quality = build_drawdown_risk_calibration_quality(
        calibration().drop(columns=["observed_drawdown_risk_rate"])
    )

    assert quality.empty


def test_drawdown_risk_calibration_quality_handles_empty_input() -> None:
    quality = build_drawdown_risk_calibration_quality(pd.DataFrame())

    assert quality.empty


def test_drawdown_risk_calibration_quality_handles_nan_heavy_input() -> None:
    data = calibration()
    data["average_probability"] = pd.NA

    quality = build_drawdown_risk_calibration_quality(data)

    assert quality.empty


def test_drawdown_risk_calibration_quality_interpretation_avoids_trading_action_words() -> None:
    rows = [
        quality_row(build_drawdown_risk_calibration_quality(calibration())),
        quality_row(
            build_drawdown_risk_calibration_quality(
                calibration(
                    predicted=(0.20, 0.40, 0.60),
                    observed=(0.40, 0.60, 0.80),
                )
            )
        ),
        quality_row(
            build_drawdown_risk_calibration_quality(
                calibration(
                    predicted=(0.40, 0.60, 0.80),
                    observed=(0.20, 0.40, 0.60),
                )
            )
        ),
        quality_row(build_drawdown_risk_calibration_quality(calibration(count=2))),
    ]
    text = " ".join(row["interpretation"] for row in rows).lower()

    forbidden_patterns = [
        r"\bbuy\b",
        r"\bsell\b",
        r"\badd\b",
        r"\breduce\b",
        r"\bincrease position\b",
        r"\bdecrease position\b",
        r"\bchange allocation\b",
        r"\bchange ranking\b",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, text) is None
