"""Decision-support report generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.portfolio.allocation import AllocationConfig, allocate_from_scores


@dataclass(frozen=True)
class DecisionReport:
    """Decision-support report in table, Markdown, and HTML-friendly forms."""

    summary: pd.DataFrame
    ticker_table: pd.DataFrame
    risk_flags: pd.DataFrame
    change_log: pd.DataFrame
    markdown: str
    html: str


def _risk_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if row.get("volatility_60d", 0) > 0.35:
        flags.append("high volatility")
    if row.get("RS vs SPY 60d", row.get("rs_spy_60d", 0)) < 0:
        flags.append("weakening relative strength")
    if row.get("ML Drawdown-Risk Probability", 0) > 0.60:
        flags.append("high drawdown probability")
    if row.get("dist_ma_50d", 0) < 0:
        flags.append("below 50dma")
    if row.get("dist_ma_200d", 0) < 0:
        flags.append("below 200dma")
    if row.get("volume_z_20d", 0) > 2:
        flags.append("abnormal volume")
    if row.get("Confidence", "Medium") == "Low":
        flags.append("model confidence low")
    return flags


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""

    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def generate_decision_report(
    current_scores: pd.DataFrame,
    latest_features: pd.DataFrame,
    benchmark_regime: str = "",
    current_weights: pd.Series | None = None,
    previous_actions: pd.Series | None = None,
    allocation_config: AllocationConfig | None = None,
) -> DecisionReport:
    """Generate a latest decision-support report."""

    cfg = allocation_config or AllocationConfig()
    scores = current_scores.copy()
    if latest_features is not None and not latest_features.empty:
        merge_columns = [column for column in latest_features.columns if column not in scores.columns or column == "Ticker"]
        scores = scores.merge(latest_features[merge_columns], on="Ticker", how="left")
    allocated = allocate_from_scores(scores, cfg, current_weights=current_weights)
    table = allocated.copy()
    table["current_price"] = table.get("Adj Close", table.get("Close"))
    table["relative_strength_rank"] = table.get("RS vs SPY 60d", table.get("rs_spy_60d")).rank(ascending=False)
    table["risk_flags"] = [", ".join(_risk_flags(row)) for _, row in table.iterrows()]

    ticker_table = table[
        [
            "Ticker",
            "current_price",
            "Rule-Based Regime",
            "ML Score",
            "ML Drawdown-Risk Probability",
            "relative_strength_rank",
            "target_weight",
            "suggested_action",
            "risk_flags",
            "allocation_explanation",
        ]
    ].rename(
        columns={
            "target_weight": "Current Target Weight",
            "suggested_action": "Suggested Action",
            "allocation_explanation": "Explanation",
        }
    )

    summary = pd.DataFrame(
        [
            {
                "Benchmark Regime": benchmark_regime,
                "Portfolio Risk State": "Defensive" if "Downtrend" in benchmark_regime else "Normal",
                "Cash Recommendation": cfg.cash_floor,
                "Gross Exposure Recommendation": min(cfg.max_gross_exposure, 1.0 - cfg.cash_floor),
            }
        ]
    )
    risk_flags = ticker_table[["Ticker", "risk_flags"]].rename(columns={"risk_flags": "Risk Flags"})
    prev = previous_actions if previous_actions is not None else pd.Series(dtype=str)
    change_rows = []
    for _, row in ticker_table.iterrows():
        previous = prev.get(row["Ticker"], "")
        current = row["Suggested Action"]
        if previous and previous != current:
            change_rows.append({"Ticker": row["Ticker"], "Previous": previous, "Current": current})
    change_log = pd.DataFrame(change_rows)
    if change_log.empty:
        change_log = pd.DataFrame([{"Ticker": "", "Previous": "", "Current": "No prior action changes supplied"}])

    markdown = "# Stock Signal Lab Decision Report\n\n"
    markdown += "## Portfolio Summary\n\n" + _markdown_table(summary) + "\n\n"
    markdown += "## Ticker Table\n\n" + _markdown_table(ticker_table) + "\n\n"
    markdown += "## Risk Flags\n\n" + _markdown_table(risk_flags) + "\n\n"
    markdown += "## Change Log\n\n" + _markdown_table(change_log) + "\n"
    html = markdown.replace("\n", "<br>\n")
    return DecisionReport(summary, ticker_table, risk_flags, change_log, markdown, html)
