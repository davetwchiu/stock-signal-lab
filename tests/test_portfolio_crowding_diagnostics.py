from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.export import build_codex_handoff, export_research_lab_diagnostics
from src.research.portfolio_crowding import build_portfolio_crowding_diagnostics


def price_frame(returns: np.ndarray, *, start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(returns) + 1, freq="B")
    price = 100.0 * pd.Series(np.r_[1.0, (1.0 + returns).cumprod()])
    return pd.DataFrame({"Date": dates, "Adj Close": price})


def synthetic_frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.01, 80)
    noise = rng.normal(0, 0.01, 80)
    return {
        "AAA": price_frame(base),
        "BBB": price_frame(0.9 * base + 0.1 * noise),
        "CCC": price_frame(0.5 * base + 0.5 * noise),
        "DDD": price_frame(rng.normal(0, 0.01, 80)),
    }


def pair_classification(pairwise: pd.DataFrame, ticker_a: str, ticker_b: str) -> str:
    row = pairwise[
        ((pairwise["ticker_a"] == ticker_a) & (pairwise["ticker_b"] == ticker_b))
        | ((pairwise["ticker_a"] == ticker_b) & (pairwise["ticker_b"] == ticker_a))
    ].iloc[0]
    return str(row["classification"])


def test_pairwise_correlation_buckets_and_crowding_summary() -> None:
    pairwise, summary, _exposure, _factor_summary = build_portfolio_crowding_diagnostics(
        synthetic_frames(),
        ["AAA", "BBB", "CCC", "DDD"],
        min_samples=20,
    )

    assert pair_classification(pairwise, "AAA", "BBB") == "high_overlap"
    assert pair_classification(pairwise, "AAA", "CCC") == "moderate_overlap"
    assert pair_classification(pairwise, "AAA", "DDD") == "diversifying"
    assert summary.loc[0, "classification"] == "high_crowding"
    assert summary.loc[0, "high_overlap_pair_count"] >= 1
    assert "Equal-weight proxy" in summary.loc[0, "reason"]


def test_insufficient_and_unavailable_samples_are_labeled() -> None:
    frames = {
        "AAA": price_frame(np.array([0.01, -0.01, 0.02])),
        "BBB": price_frame(np.array([0.01, -0.01, 0.02])),
    }
    pairwise, summary, _exposure, _factor_summary = build_portfolio_crowding_diagnostics(
        frames,
        ["AAA", "BBB", "MISSING"],
        min_samples=20,
    )

    classes = set(pairwise["classification"])
    assert "insufficient_sample" in classes
    assert "unavailable" in classes
    assert summary.loc[0, "classification"] == "insufficient_sample"


def test_factor_proxy_mapping_is_conservative() -> None:
    _pairwise, _summary, exposure, factor_summary = build_portfolio_crowding_diagnostics(
        {},
        ["NVDA", "SMH", "ZZZ"],
    )

    indexed = exposure.set_index("ticker")
    assert indexed.loc["NVDA", "classification"] == "mapped"
    assert indexed.loc["SMH", "classification"] == "proxy_only"
    assert indexed.loc["ZZZ", "classification"] == "unknown"
    assert "crowded" in set(factor_summary["classification"])
    assert "unknown" in set(factor_summary["classification"])


def test_portfolio_crowding_tables_export_and_handoff(tmp_path: Path) -> None:
    pairwise, summary, exposure, factor_summary = build_portfolio_crowding_diagnostics(
        synthetic_frames(),
        ["AAA", "BBB", "CCC", "DDD"],
    )

    result = export_research_lab_diagnostics(
        run_metadata={"created_at": "2026-06-17T10:00:00", "ticker_count": 4},
        tables={
            "portfolio_correlation_diagnostics": pairwise,
            "portfolio_crowding_summary": summary,
            "portfolio_factor_proxy_exposure": exposure,
            "portfolio_factor_crowding_summary": factor_summary,
        },
        output_root=tmp_path,
        run_id="run",
    )

    assert result.manifest["row_counts"]["portfolio_correlation_diagnostics.csv"] == len(pairwise)
    assert result.manifest["row_counts"]["portfolio_crowding_summary.csv"] == 1
    assert (result.run_dir / "portfolio_factor_proxy_exposure.csv").exists()
    assert "## Portfolio crowding diagnostics" in result.codex_handoff


def test_codex_handoff_summarizes_portfolio_crowding() -> None:
    handoff = build_codex_handoff(
        run_metadata={"created_at": "2026-06-17T10:00:00", "ticker_count": 3},
        tables={
            "portfolio_crowding_summary": pd.DataFrame(
                {
                    "classification": ["moderate_crowding"],
                    "high_overlap_pair_count": [1],
                    "largest_cluster_size": [2],
                }
            ),
            "portfolio_factor_crowding_summary": pd.DataFrame(
                {"proxy_group": ["semiconductor"], "classification": ["watch"]}
            ),
        },
        manifest={"files_written": []},
    )

    assert "Correlation crowding: moderate_crowding" in handoff
    assert "Factor proxy rows=1: watch=1." in handoff
    assert "Risk-visibility only, not alpha" in handoff
    assert "one large bet" in handoff
