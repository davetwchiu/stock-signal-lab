"""Decision Cockpit table and explanation builders."""

from __future__ import annotations

import pandas as pd

from src.decision.config import DecisionConfig, DecisionProfile
from src.portfolio.allocation import AllocationConfig, allocate_from_scores


ACTION_ORDER = ("Add", "Hold", "Trim", "Exit", "Watch")


def target_exposure_bucket(target_weight: float, max_position_size: float) -> str:
    """Convert a target weight into a 0/25/50/75/100 exposure bucket."""

    if max_position_size <= 0 or pd.isna(target_weight):
        return "0%"
    ratio = max(0.0, min(1.0, float(target_weight) / max_position_size))
    if ratio >= 0.875:
        bucket = 100
    elif ratio >= 0.625:
        bucket = 75
    elif ratio >= 0.375:
        bucket = 50
    elif ratio >= 0.125:
        bucket = 25
    else:
        bucket = 0
    return f"{bucket}%"


def confidence_from_score(score: float, drawdown_risk_probability: float) -> str:
    """Confidence bucket for Decision Mode."""

    if pd.isna(score) or pd.isna(drawdown_risk_probability):
        return "Low"
    distance = max(abs(float(score) - 50.0) / 50.0, abs(float(drawdown_risk_probability) - 0.5) * 2)
    if distance >= 0.50:
        return "High"
    if distance >= 0.20:
        return "Medium"
    return "Low"


def one_line_reason(row: pd.Series) -> str:
    """Generate a concise reason for the decision table."""

    action = row.get("Suggested action", row.get("Suggested Action", "Watch"))
    regime = str(row.get("Rule-based regime", row.get("Rule-Based Regime", "")))
    score = row.get("ML score", row.get("ML Score", pd.NA))
    risk = row.get("Drawdown-risk probability", row.get("ML Drawdown-Risk Probability", pd.NA))
    rs_rank = row.get("Relative strength rank", pd.NA)

    if action == "Add":
        return "Strong score with acceptable drawdown risk and supportive trend."
    if action == "Hold":
        return "Evidence is constructive, but risk or conviction does not justify adding."
    if action == "Trim":
        return "Target exposure is lower than current exposure because risk has risen."
    if action == "Exit":
        return "Risk controls call for no exposure under the current setup."
    if "Downtrend" in regime:
        return "Trend regime is weak, so the ticker stays on watch."
    if pd.notna(score) and score < 40:
        return "ML score is not strong enough for fresh exposure."
    if pd.notna(risk) and risk >= 0.60:
        return "Drawdown-risk probability is too high for new exposure."
    if pd.notna(rs_rank):
        return "Waiting for stronger relative strength or lower drawdown risk."
    return "Insufficient conviction for active exposure."


def build_decision_table(
    current_scores: pd.DataFrame,
    latest_features: pd.DataFrame,
    config: DecisionConfig,
    profile: DecisionProfile,
    current_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the simplified one-row-per-ticker Decision Cockpit table."""

    if current_scores.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Price",
                "Rule-based regime",
                "ML score",
                "Drawdown-risk probability",
                "Relative strength rank",
                "Suggested action",
                "Target exposure bucket",
                "Confidence",
                "One-line reason",
            ]
        )

    scores = current_scores.copy()
    if latest_features is not None and not latest_features.empty:
        merge_columns = [column for column in latest_features.columns if column not in scores.columns or column == "Ticker"]
        scores = scores.merge(latest_features[merge_columns], on="Ticker", how="left")

    allocation_config = AllocationConfig(
        max_position_size=profile.max_single_position_exposure,
        cash_floor=profile.cash_floor,
        max_gross_exposure=1.0 - profile.cash_floor,
        drawdown_risk_threshold=profile.high_drawdown_risk_threshold,
        moderate_drawdown_risk_threshold=profile.moderate_drawdown_risk_threshold,
    )
    allocated = allocate_from_scores(scores, allocation_config, current_weights=current_weights)

    rs_column = "rs_spy_60d" if "rs_spy_60d" in allocated else "RS vs SPY 60d"
    allocated["Relative strength rank"] = allocated.get(rs_column, pd.Series(index=allocated.index, dtype=float)).rank(
        ascending=False,
        method="min",
    )
    output = pd.DataFrame(
        {
            "Ticker": allocated["Ticker"],
            "Price": allocated.get("Adj Close", allocated.get("Close")),
            "Rule-based regime": allocated.get("Rule-Based Regime", allocated.get("regime", "")),
            "ML score": allocated["ML Score"].round(0),
            "Drawdown-risk probability": allocated["ML Drawdown-Risk Probability"],
            "Relative strength rank": allocated["Relative strength rank"],
            "Suggested action": allocated["suggested_action"],
            "Target exposure bucket": [
                target_exposure_bucket(weight, profile.max_single_position_exposure)
                for weight in allocated["target_weight"]
            ],
            "Confidence": [
                confidence_from_score(score, risk)
                for score, risk in zip(allocated["ML Score"], allocated["ML Drawdown-Risk Probability"], strict=False)
            ],
        }
    )
    output["One-line reason"] = [one_line_reason(row) for _, row in output.iterrows()]
    return output.sort_values(["Suggested action", "ML score"], ascending=[True, False]).reset_index(drop=True)


def action_counts(decision_table: pd.DataFrame) -> dict[str, int]:
    """Count Add/Hold/Trim/Exit/Watch labels."""

    counts = decision_table.get("Suggested action", pd.Series(dtype=str)).value_counts().to_dict()
    return {action: int(counts.get(action, 0)) for action in ACTION_ORDER}

