from __future__ import annotations

import pandas as pd
import pytest

from src.ml.diagnostics import build_ml_reliability_by_regime, build_ml_score_regime_bucket_audit
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
