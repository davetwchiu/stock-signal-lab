"""Markdown report generation for the simplified Decision Cockpit."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.decision.table import action_counts


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def portfolio_summary_text(
    market_regime: str,
    decision_table: pd.DataFrame,
    suggested_cash_level: float,
) -> str:
    """Generate a plain-English portfolio summary."""

    counts = action_counts(decision_table)
    high_risk = int((decision_table.get("Drawdown-risk probability", pd.Series(dtype=float)) >= 0.60).sum())
    if "Downtrend" in market_regime or high_risk >= max(2, len(decision_table) // 3):
        risk_state = "elevated"
        posture = "prioritize cash, avoid broad aggressive adding, and trim weak names."
    elif counts["Add"] > counts["Trim"] + counts["Exit"]:
        risk_state = "constructive"
        posture = "selective adds are acceptable while respecting position-size limits."
    else:
        risk_state = "moderate"
        posture = "keep core exposure, avoid broad aggressive adding, and wait for clearer leadership."
    return (
        f"Risk state is {risk_state}. Market regime is {market_regime or 'not available'}. "
        f"Suggested cash level is {suggested_cash_level:.0%}; {posture}"
    )


def main_warning(decision_table: pd.DataFrame, market_regime: str) -> str:
    """Return the main cockpit warning, if any."""

    if "Downtrend" in market_regime:
        return "Benchmark trend is weak; broad de-risking may be appropriate."
    if decision_table.empty:
        return "No decision rows available."
    high_risk = decision_table[decision_table["Drawdown-risk probability"] >= 0.60]
    if not high_risk.empty:
        return f"{len(high_risk)} names have elevated drawdown-risk probability."
    low_conf = decision_table[decision_table["Confidence"] == "Low"]
    if len(low_conf) >= max(2, len(decision_table) // 3):
        return "Many recommendations have low confidence."
    return "No dominant warning."


def generate_markdown_report(
    decision_table: pd.DataFrame,
    profile: str,
    benchmark: str,
    market_regime: str,
    suggested_gross_exposure: float,
    suggested_cash_level: float,
    summary_text: str,
    previous_actions: pd.Series | None = None,
    report_date: date | None = None,
) -> str:
    """Generate a readable Markdown decision report."""

    previous = previous_actions if previous_actions is not None else pd.Series(dtype=str)
    upgraded: list[str] = []
    downgraded: list[str] = []
    action_rank = {"Watch": 0, "Exit": 0, "Trim": 1, "Hold": 2, "Add": 3}
    for _, row in decision_table.iterrows():
        old = previous.get(row["Ticker"])
        new = row["Suggested action"]
        if old is None or old == new:
            continue
        if action_rank.get(new, 0) > action_rank.get(old, 0):
            upgraded.append(row["Ticker"])
        else:
            downgraded.append(row["Ticker"])

    actions = decision_table[
        [
            "Ticker",
            "Suggested action",
            "Target exposure bucket",
            "Confidence",
            "One-line reason",
        ]
    ]
    risks = decision_table[
        ["Ticker", "Drawdown-risk probability", "Rule-based regime", "One-line reason"]
    ].sort_values("Drawdown-risk probability", ascending=False)

    today = report_date or date.today()
    lines = [
        "# Stock Signal Lab Decision Report",
        "",
        f"Date: {today.isoformat()}",
        f"Profile: {profile}",
        f"Benchmark: {benchmark}",
        f"Market regime: {market_regime}",
        f"Suggested gross exposure: {suggested_gross_exposure:.0%}",
        f"Suggested cash level: {suggested_cash_level:.0%}",
        "",
        "## Summary",
        summary_text,
        "",
        "## Actions",
        _markdown_table(actions),
        "",
        "## Key upgrades",
        ", ".join(upgraded) if upgraded else "None supplied or no upgrades.",
        "",
        "## Key downgrades",
        ", ".join(downgraded) if downgraded else "None supplied or no downgrades.",
        "",
        "## Main risks",
        _markdown_table(risks.head(8)),
        "",
        "_Research and decision support only. Not financial advice, not a price target, and not an automatic trading system._",
    ]
    return "\n".join(lines)

