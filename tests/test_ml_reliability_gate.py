from __future__ import annotations

import pandas as pd

from src.ml.diagnostics import build_ml_reliability_gate_diagnostics
from src.research.export import export_research_lab_diagnostics


def gate_panel(
    *,
    inverted_regime_returns: tuple[float, float, float] = (0.01, -0.02, -0.08),
    rows_per_bucket: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=rows_per_bucket * 6, freq="B")
    row_index = 0
    regime_specs = (
        (
            "Helpful regime",
            (0.00, 0.04, 0.10),
            (0, 1, 1),
            (0, 0, 0),
            (0.10, 0.20, 0.20),
        ),
        (
            "Inverted regime",
            inverted_regime_returns,
            (1, 1, 0),
            (0, 1, 1),
            (0.20, 0.40, 0.60),
        ),
    )
    for regime, returns, labels, risks, risk_probabilities in regime_specs:
        for bucket, score, probability, forward_excess_return, actual_out, actual_risk, probability_risk in zip(
            ("low", "medium", "high"),
            (20.0, 55.0, 85.0),
            (0.20, 0.55, 0.85),
            returns,
            labels,
            risks,
            risk_probabilities,
            strict=True,
        ):
            for bucket_index in range(rows_per_bucket):
                date = dates[row_index]
                ticker = f"{regime[:1]}{bucket[:1]}{bucket_index:02d}"
                rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "ML Score": score,
                        "probability_out": probability,
                        "actual_out": actual_out,
                        "forward_excess_return": forward_excess_return,
                        "probability_risk": probability_risk,
                        "actual_risk": actual_risk,
                    }
                )
                baseline_rows.append({"Date": date, "Ticker": ticker, "regime": regime})
                row_index += 1
    return pd.DataFrame(rows), pd.DataFrame(baseline_rows)


def diagnostics_by_gate(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index("gate_name")


def test_reliability_gate_reports_no_gate_baseline_metrics() -> None:
    panel, baseline = gate_panel()

    diagnostics = diagnostics_by_gate(
        build_ml_reliability_gate_diagnostics(panel, baseline_panel=baseline, min_samples=10, min_bucket_size=3)
    )

    baseline_row = diagnostics.loc["no_gate_baseline"]
    assert baseline_row["sample_count"] == 60
    assert baseline_row["pass_count"] == 60
    assert baseline_row["fail_count"] == 0
    assert baseline_row["retention_rate"] == 1.0
    assert baseline_row["classification"] == "mixed"


def test_reliability_gate_classifies_useful_gate() -> None:
    panel, baseline = gate_panel()

    diagnostics = diagnostics_by_gate(
        build_ml_reliability_gate_diagnostics(panel, baseline_panel=baseline, min_samples=10, min_bucket_size=3)
    )

    gate = diagnostics.loc["exclude_inverted_regimes"]
    assert gate["classification"] == "useful"
    assert gate["improvement_flag"]
    assert gate["pass_avg_forward_excess_return"] > diagnostics.loc[
        "no_gate_baseline", "pass_avg_forward_excess_return"
    ]
    assert gate["pass_drawdown_rate"] < diagnostics.loc["no_gate_baseline", "pass_drawdown_rate"]


def test_reliability_gate_classifies_harmful_inverted_gate() -> None:
    panel, baseline = gate_panel(inverted_regime_returns=(0.20, 0.15, 0.10))

    diagnostics = diagnostics_by_gate(
        build_ml_reliability_gate_diagnostics(panel, baseline_panel=baseline, min_samples=10, min_bucket_size=3)
    )

    gate = diagnostics.loc["exclude_inverted_regimes"]
    assert gate["classification"] == "harmful"
    assert gate["pass_avg_forward_excess_return"] < diagnostics.loc[
        "no_gate_baseline", "pass_avg_forward_excess_return"
    ]


def test_reliability_gate_handles_insufficient_samples() -> None:
    panel, baseline = gate_panel(rows_per_bucket=2)

    diagnostics = diagnostics_by_gate(
        build_ml_reliability_gate_diagnostics(panel, baseline_panel=baseline, min_samples=10, min_bucket_size=3)
    )

    gate = diagnostics.loc["exclude_inverted_regimes"]
    assert gate["classification"] == "insufficient_sample"
    assert gate["insufficient_sample_flag"]


def test_reliability_gate_handles_missing_columns_as_unavailable() -> None:
    panel, baseline = gate_panel()

    diagnostics = build_ml_reliability_gate_diagnostics(
        panel.drop(columns=["probability_risk"]),
        baseline_panel=baseline,
        min_samples=10,
        min_bucket_size=3,
    )

    assert set(diagnostics["classification"]) == {"unavailable"}
    assert diagnostics["reason"].str.contains("probability_risk").all()


def test_reliability_gate_export_writes_csv(tmp_path) -> None:
    panel, baseline = gate_panel()
    table = build_ml_reliability_gate_diagnostics(
        panel,
        baseline_panel=baseline,
        min_samples=10,
        min_bucket_size=3,
    )

    result = export_research_lab_diagnostics(
        run_metadata={"created_at": "2026-06-16T10:13:30", "ticker_count": 2},
        tables={"ml_reliability_gate_diagnostics": table},
        output_root=tmp_path,
        run_id="run",
    )

    exported = pd.read_csv(result.run_dir / "ml_reliability_gate_diagnostics.csv")
    assert "classification" in exported.columns
    assert result.manifest["row_counts"]["ml_reliability_gate_diagnostics.csv"] == len(table)
