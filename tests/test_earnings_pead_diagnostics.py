from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ml.diagnostics import build_earnings_pead_diagnostics
from src.research.earnings_events import load_earnings_events
from src.research.export import export_research_lab_diagnostics


def synthetic_score_panel(post_return: float = 0.03, post_drawdown: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=140)
    event_dates = pd.Series([dates[20], dates[60], dates[100]])
    rows: list[dict[str, object]] = []
    for date in dates:
        in_post_window = any(0 <= (date.normalize() - event_date.normalize()).days <= 20 for event_date in event_dates)
        rows.append(
            {
                "Date": date,
                "Ticker": "AAA",
                "ML Score": 80.0 if in_post_window else 55.0,
                "actual_out": 1 if in_post_window and post_return > 0 else 0,
                "actual_risk": post_drawdown if in_post_window else 0,
                "forward_excess_return": post_return if in_post_window else 0.0,
                "forward_drawdown": -0.12 if in_post_window and post_drawdown else -0.03,
                "Open": 101.0,
                "Close": 100.0,
                "Adj Close": 100.0 + len(rows),
            }
        )
    events = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "earnings_date": event_dates,
            "report_timing": ["before_open", "after_close", ""],
        }
    )
    events.attrs["earnings_event_status"] = "loaded"
    events.attrs["earnings_event_reason"] = "Loaded test events."
    return pd.DataFrame(rows), events


def test_load_earnings_events_parses_valid_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "earnings_events.csv"
    csv_path.write_text(
        "ticker,earnings_date,report_timing,eps_surprise_pct\n"
        " aaa ,2024-02-01,Before_Open,4.5\n"
        "AAA,2024-02-01,after_close,5.0\n"
        "bbb,2024-03-04,unexpected,\n",
        encoding="utf-8",
    )

    events = load_earnings_events(csv_path)

    assert events["ticker"].tolist() == ["AAA", "BBB"]
    assert events.loc[events["ticker"] == "AAA", "report_timing"].iloc[0] == "after_close"
    assert events.loc[events["ticker"] == "BBB", "report_timing"].iloc[0] == ""
    assert events.attrs["earnings_event_status"] == "loaded"


def test_missing_earnings_event_csv_exports_unavailable_classification(tmp_path: Path) -> None:
    missing = load_earnings_events(root=tmp_path)
    panel, _events = synthetic_score_panel()

    _event_table, _window_table, summary = build_earnings_pead_diagnostics(panel, missing, baseline_panel=panel)

    assert missing.attrs["earnings_event_status"] == "missing"
    assert summary.loc[0, "classification"] == "unavailable"
    assert summary.loc[0, "pead_signal_direction"] == "unavailable"


def test_insufficient_event_data_is_classified_conservatively() -> None:
    panel, events = synthetic_score_panel()
    one_event = events.iloc[[0]].copy()
    one_event.attrs["earnings_event_status"] = "loaded"
    one_event.attrs["earnings_event_reason"] = "Loaded test events."

    event_table, _window_table, summary = build_earnings_pead_diagnostics(panel, one_event, baseline_panel=panel)

    assert event_table.loc[0, "usable_event_count"] == 1
    assert summary.loc[0, "classification"] == "insufficient_event_data"


def test_post_earnings_window_counts_are_derived_from_event_dates() -> None:
    panel, events = synthetic_score_panel()

    event_table, window_table, _summary = build_earnings_pead_diagnostics(panel, events, baseline_panel=panel)

    indexed = window_table.set_index("earnings_window")
    assert event_table.loc[0, "pre_earnings_sample_count"] > 0
    assert indexed.loc["post_earnings_5d", "sample_count"] > 0
    assert indexed.loc["post_earnings_20d", "sample_count"] > indexed.loc["post_earnings_5d", "sample_count"]


@pytest.mark.parametrize(
    ("post_return", "post_drawdown", "expected"),
    [
        (0.03, 0, "useful"),
        (0.001, 0, "mixed"),
        (-0.02, 0, "harmful"),
        (0.03, 1, "harmful"),
    ],
)
def test_pead_classification_uses_return_and_drawdown_evidence(
    post_return: float,
    post_drawdown: int,
    expected: str,
) -> None:
    panel, events = synthetic_score_panel(post_return=post_return, post_drawdown=post_drawdown)

    _event_table, _window_table, summary = build_earnings_pead_diagnostics(panel, events, baseline_panel=panel)

    assert summary.loc[0, "classification"] == expected


def test_earnings_pead_tables_are_exported_in_research_bundle(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "sample_count": [20],
            "event_count": [3],
            "usable_event_count": [3],
            "post_earnings_positive_rate": [0.6],
            "post_earnings_avg_forward_excess_return": [0.03],
            "post_earnings_drawdown_rate": [0.1],
            "pead_signal_direction": ["positive"],
            "ml_near_earnings_effect": ["supportive"],
            "classification": ["useful"],
            "reason": ["Synthetic export test."],
        }
    )

    result = export_research_lab_diagnostics(
        run_metadata={"benchmark": "QQQ", "ticker_count": 1},
        tables={"earnings_pead_summary": summary},
        output_root=tmp_path,
        run_id="run",
    )

    assert (result.run_dir / "earnings_pead_summary.csv").exists()
    assert result.manifest["row_counts"]["earnings_pead_summary.csv"] == 1
    assert "## Earnings / PEAD diagnostics" in result.codex_handoff
    assert "classifications: useful=1" in result.codex_handoff
