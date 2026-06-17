from __future__ import annotations

import pandas as pd
import pytest

from src.ml.diagnostics import (
    build_drawdown_risk_prevalence_baseline_comparison,
    build_drawdown_risk_regime_calibration,
    build_ml_reliability_by_regime,
    build_ml_score_regime_bucket_audit,
)
from src.research.export import export_research_lab_diagnostics


def reliability_panel(
    *,
    regime: str = "Uptrend / low volatility",
    direction: str = "reliable",
    rows_per_bucket: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=rows_per_bucket * 3, freq="B")
    row_index = 0
    for bucket, score, probability in (
        ("low", 20.0, 0.20),
        ("medium", 55.0, 0.55),
        ("high", 85.0, 0.85),
    ):
        for bucket_index in range(rows_per_bucket):
            if direction == "inverted":
                actual_out = int(bucket == "low" or (bucket == "medium" and bucket_index % 2 == 0))
                forward_excess_return = {"low": 0.08, "medium": 0.03, "high": 0.00}[bucket]
            else:
                actual_out = int(bucket == "high" or (bucket == "medium" and bucket_index % 2 == 0))
                forward_excess_return = {"low": 0.00, "medium": 0.03, "high": 0.08}[bucket]
            date = dates[row_index]
            ticker = f"{bucket[:1].upper()}{bucket_index:02d}"
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "ML Score": score,
                    "probability_out": probability,
                    "actual_out": actual_out,
                    "forward_excess_return": forward_excess_return,
                    "probability_risk": 0.10,
                    "actual_risk": 0,
                }
            )
            baseline_rows.append({"Date": date, "Ticker": ticker, "regime": regime})
            row_index += 1
    return pd.DataFrame(rows), pd.DataFrame(baseline_rows)


def reliability_by_regime(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index("regime")


def risk_calibration_panel(rows_per_bucket: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=rows_per_bucket * 3, freq="B")
    row_index = 0
    for bucket, ticker, probability in (
        ("low", "LOW", 0.10),
        ("mid", "MID", 0.50),
        ("high", "HIGH", 0.90),
    ):
        for bucket_index in range(rows_per_bucket):
            actual_risk = int(bucket == "high" or (bucket == "mid" and bucket_index % 2 == 0))
            date = dates[row_index]
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "fold": 1 + (row_index % 3),
                    "probability_risk": probability,
                    "actual_risk": actual_risk,
                }
            )
            baseline_rows.append({"Date": date, "Ticker": ticker, "regime": "Uptrend / low volatility"})
            row_index += 1
    return pd.DataFrame(rows), pd.DataFrame(baseline_rows)


def test_drawdown_risk_regime_calibration_classifies_usable_signal() -> None:
    panel, baseline = risk_calibration_panel()

    diagnostics = build_drawdown_risk_regime_calibration(
        panel,
        baseline_panel=baseline,
        min_samples=10,
        min_class_count=3,
        min_bucket_size=3,
    ).set_index("regime")

    row = diagnostics.loc["Uptrend / low volatility"]
    assert row["sample_count"] == 30
    assert row["ticker_count"] == 3
    assert row["fold_count"] == 3
    assert row["event_prevalence"] == pytest.approx(0.5)
    assert row["calibration_gap"] == pytest.approx(0.0)
    assert row["pr_auc"] > row["event_prevalence"]
    assert row["bucket_spread"] == pytest.approx(1.0)
    assert row["monotonicity"] == "aligned"
    assert row["worst_ticker"] == "MID"
    assert row["classification"] == "usable_risk_signal"


def test_drawdown_risk_regime_calibration_export_writes_csv(tmp_path) -> None:
    panel, baseline = risk_calibration_panel()
    table = build_drawdown_risk_regime_calibration(
        panel,
        baseline_panel=baseline,
        min_samples=10,
        min_class_count=3,
        min_bucket_size=3,
    )

    result = export_research_lab_diagnostics(
        run_metadata={"created_at": "2026-06-17T10:13:30", "ticker_count": 3},
        tables={"drawdown_risk_regime_calibration": table},
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "drawdown_risk_regime_calibration.csv")
    assert "classification" in exported.columns
    assert result.manifest["row_counts"]["drawdown_risk_regime_calibration.csv"] == len(table)


def test_drawdown_risk_prevalence_baseline_uses_fold_training_rates() -> None:
    predictions = pd.DataFrame(
        {
            "fold": [1, 1, 1, 1, 2, 2, 2, 2],
            "Date": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-03-04",
                    "2024-03-05",
                    "2024-03-06",
                    "2024-06-03",
                    "2024-06-04",
                    "2024-06-05",
                    "2024-06-06",
                ]
            ),
            "Ticker": ["AAA", "BBB", "CCC", "DDD", "AAA", "BBB", "CCC", "DDD"],
            "actual": [0, 0, 1, 1, 1, 1, 1, 0],
            "probability": [0.90, 0.90, 0.10, 0.10, 0.10, 0.10, 0.10, 0.90],
        }
    )
    baseline = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-04-01",
                    "2024-04-02",
                    "2024-04-03",
                    "2024-04-04",
                    "2024-03-01",
                    "2024-03-04",
                    "2024-03-05",
                    "2024-03-06",
                    "2024-06-03",
                    "2024-06-04",
                    "2024-06-05",
                    "2024-06-06",
                ]
            ),
            "Ticker": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "AAA",
                "BBB",
                "CCC",
                "DDD",
            ],
            "label_drawdown_risk_20d": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
            "regime": [
                "calm",
                "calm",
                "thin",
                "thin",
                "calm",
                "calm",
                "thin",
                "thin",
                "calm",
                "calm",
                "thin",
                "thin",
                "calm",
                "calm",
                "thin",
                "thin",
            ],
        }
    )
    fold_details = pd.DataFrame(
        {
            "fold": [1, 2],
            "train_start": ["2024-01-01", "2024-04-01"],
            "train_end": ["2024-01-31", "2024-04-30"],
        }
    )

    table = build_drawdown_risk_prevalence_baseline_comparison(
        predictions,
        baseline,
        fold_details,
        min_regime_train_samples=3,
        min_regime_train_events=2,
        min_bucket_size=1,
    ).set_index("comparator")

    assert table.loc["model_predicted_risk", "sample_count"] == 8
    assert table.loc["global_fold_prevalence_baseline", "mean_predicted_risk"] == pytest.approx(0.5)
    assert table.loc["regime_fold_prevalence_baseline", "fallback_count"] == 8
    assert table.loc["model_predicted_risk", "classification"] == "baseline_beats_model"
    assert table.loc["global_fold_prevalence_baseline", "classification"] == "baseline_beats_model"
    assert table.loc["global_fold_prevalence_baseline", "fold_train_prevalence_details"] == "1:4:0.000000;2:4:1.000000"


def test_ml_reliability_by_regime_classifies_reliable_regime() -> None:
    panel, baseline = reliability_panel()

    diagnostics = reliability_by_regime(build_ml_reliability_by_regime(panel, baseline_panel=baseline))

    row = diagnostics.loc["Uptrend / low volatility"]
    assert row["sample_count"] == 30
    assert row["positive_rate"] == pytest.approx(0.5)
    assert row["roc_auc"] > 0.90
    assert row["pr_auc"] > row["positive_rate"]
    assert row["bucket_spread"] == pytest.approx(0.08)
    assert row["score_bucket_monotonicity"] == "aligned"
    assert row["classification"] == "reliable"
    assert not row["inversion_flag"]
    assert not row["insufficient_sample_flag"]


def test_ml_reliability_by_regime_handles_insufficient_sample() -> None:
    panel, baseline = reliability_panel(rows_per_bucket=2)

    diagnostics = reliability_by_regime(build_ml_reliability_by_regime(panel, baseline_panel=baseline))

    row = diagnostics.loc["Uptrend / low volatility"]
    assert row["classification"] == "insufficient_sample"
    assert row["insufficient_sample_flag"]
    assert pd.isna(row["bucket_spread"])


def test_ml_reliability_by_regime_flags_inverted_regime() -> None:
    panel, baseline = reliability_panel(regime="Downtrend / high risk", direction="inverted")

    diagnostics = reliability_by_regime(build_ml_reliability_by_regime(panel, baseline_panel=baseline))

    row = diagnostics.loc["Downtrend / high risk"]
    assert row["classification"] == "inverted"
    assert row["inversion_flag"]
    assert row["bucket_spread"] < 0
    assert row["score_bucket_monotonicity"] == "inverted"


def test_ml_score_regime_bucket_audit_flags_uptrend_overextension_risk() -> None:
    panel, baseline = reliability_panel(regime="Uptrend / high volatility", direction="inverted")
    panel.loc[panel["ML Score"].eq(85.0), "actual_risk"] = 1

    audit = build_ml_score_regime_bucket_audit(
        panel,
        baseline_panel=baseline,
        min_samples=10,
        min_bucket_size=3,
    ).set_index("regime")

    row = audit.loc["Uptrend / high volatility"]
    assert row["sample_count"] == 30
    assert row["ticker_count"] == 30
    assert row["high_score_sample_count"] == 10
    assert row["low_score_sample_count"] == 10
    assert row["opportunity_bucket_spread"] < 0
    assert row["drawdown_reversal_bucket_spread"] > 0
    assert row["inversion_flag"]
    assert row["overextension_risk_flag"]
    assert row["classification"] == "overextension_risk"
    assert row["recommended_decision"] == "Pivot"


def test_ml_score_regime_bucket_audit_export_writes_csv(tmp_path) -> None:
    panel, baseline = reliability_panel(regime="Uptrend / high volatility", direction="inverted")
    table = build_ml_score_regime_bucket_audit(
        panel,
        baseline_panel=baseline,
        min_samples=10,
        min_bucket_size=3,
    )

    result = export_research_lab_diagnostics(
        run_metadata={"created_at": "2026-06-17T10:13:30", "ticker_count": 1},
        tables={"ml_score_regime_bucket_audit": table},
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "ml_score_regime_bucket_audit.csv")
    assert "recommended_decision" in exported.columns
    assert result.manifest["row_counts"]["ml_score_regime_bucket_audit.csv"] == len(table)


def test_ml_reliability_by_regime_handles_missing_columns_and_empty_data() -> None:
    panel, baseline = reliability_panel()

    assert build_ml_reliability_by_regime(pd.DataFrame()).empty
    assert build_ml_reliability_by_regime(
        panel.drop(columns=["probability_risk"]),
        baseline_panel=baseline,
    ).empty
    assert build_ml_reliability_by_regime(panel, baseline_panel=baseline.drop(columns=["regime"])).empty
