"""Plain-English ticker explanations for Decision Mode."""

from __future__ import annotations

import pandas as pd


def bullish_evidence(row: pd.Series) -> list[str]:
    """Return short bullish evidence bullets."""

    evidence: list[str] = []
    if row.get("dist_ma_50d", 0) > 0:
        evidence.append("Price remains above the 50-day moving average.")
    if row.get("dist_ma_200d", 0) > 0:
        evidence.append("Price remains above the 200-day moving average.")
    if row.get("rs_qqq_60d", row.get("rs_spy_60d", 0)) > 0:
        evidence.append("Relative strength versus the benchmark is positive.")
    if row.get("ML Outperformance Probability", 0) >= 0.60:
        evidence.append("ML outperformance probability is above the neutral zone.")
    if not evidence:
        evidence.append("There is not much bullish evidence under the current rules.")
    return evidence[:3]


def bearish_evidence(row: pd.Series) -> list[str]:
    """Return short risk flag bullets."""

    flags: list[str] = []
    if row.get("ML Drawdown-Risk Probability", 0) >= 0.60:
        flags.append("Drawdown-risk probability is elevated.")
    if row.get("volatility_60d", 0) > 0.35:
        flags.append("Short-term volatility is elevated.")
    if row.get("dist_ma_50d", 0) < 0:
        flags.append("Price is below the 50-day moving average.")
    if row.get("dist_ma_200d", 0) < 0:
        flags.append("Price is below the 200-day moving average.")
    if row.get("volume_z_20d", 0) > 2:
        flags.append("Recent volume is unusually high.")
    if not flags:
        flags.append("No major risk flag dominates the current recommendation.")
    return flags[:3]


def recommendation_change_triggers(row: pd.Series) -> list[str]:
    """Explain what would change the current action."""

    action = row.get("Suggested action", row.get("Suggested Action", "Watch"))
    if action in {"Add", "Hold"}:
        return [
            "Downgrade if price breaks below the 50-day moving average.",
            "Downgrade if ML score falls below the hold zone.",
            "Trim if drawdown-risk probability rises above the high-risk threshold.",
        ]
    return [
        "Upgrade if drawdown-risk probability falls and relative strength improves.",
        "Upgrade if price recovers above key moving averages.",
        "Upgrade if ML score moves back into the constructive zone.",
    ]


def ticker_explanation(decision_row: pd.Series, feature_row: pd.Series) -> dict[str, object]:
    """Build a concise Explain / Why payload for one ticker."""

    merged = feature_row.copy()
    for key, value in decision_row.items():
        merged[key] = value
    action = merged.get("Suggested action", "Watch")
    bucket = merged.get("Target exposure bucket", "0%")
    reason = merged.get("One-line reason", "")
    return {
        "ticker": merged.get("Ticker", ""),
        "action": action,
        "target_exposure": bucket,
        "reason": reason,
        "bullish_evidence": bullish_evidence(merged),
        "bearish_evidence": bearish_evidence(merged),
        "what_would_change": recommendation_change_triggers(merged),
    }

