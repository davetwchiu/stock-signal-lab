"""Decision Cockpit table and explanation builders."""

from __future__ import annotations

import re

import pandas as pd

from src.decision.config import DecisionConfig, DecisionProfile
from src.portfolio.allocation import AllocationConfig, allocate_from_scores


ACTION_ORDER = ("Add", "Hold", "Trim", "Exit", "Watch")
ACTION_RANK = {action: index for index, action in enumerate(ACTION_ORDER)}


def parse_current_weights_input(raw: str) -> pd.Series:
    """Parse optional ticker weights from sidebar text."""

    text = str(raw or "").strip()
    if not text:
        return pd.Series(dtype=float)

    weights: dict[str, float] = {}
    entries = [part.strip() for part in re.split(r"[\n,;]+", text) if part.strip()]
    for entry in entries:
        if "=" in entry:
            ticker, value = [part.strip() for part in entry.split("=", maxsplit=1)]
        else:
            pieces = entry.split()
            if len(pieces) != 2:
                raise ValueError(f"Use TICKER weight pairs, for example NVDA 0.12: {entry}")
            ticker, value = pieces
        ticker = ticker.upper()
        if not ticker:
            raise ValueError("Ticker is required for each current weight.")
        try:
            weights[ticker] = float(value)
        except ValueError as error:
            raise ValueError(f"Weight for {ticker} must be numeric.") from error
    return pd.Series(weights, dtype=float)


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


def _clean_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_phrase(score: object) -> str:
    value = _clean_float(score)
    if value is None:
        return "ML score unavailable"
    rounded = round(value)
    if value >= 80:
        level = "very high"
    elif value >= 60:
        level = "constructive"
    elif value >= 40:
        level = "mixed"
    elif value >= 20:
        level = "weak"
    else:
        level = "very weak"
    return f"{level} ML score {rounded:.0f}"


def _risk_phrase(risk: object) -> str:
    value = _clean_float(risk)
    if value is None:
        return "drawdown risk unavailable"
    if value >= 0.60:
        level = "high"
    elif value >= 0.40:
        level = "elevated"
    elif value >= 0.25:
        level = "moderate"
    else:
        level = "low"
    return f"{level} drawdown risk {value:.0%}"


def _regime_phrase(regime: object) -> str:
    text = str(regime).strip()
    if not text:
        return "trend regime unavailable"
    lowered = text.lower()
    if "downtrend" in lowered:
        return "weak downtrend regime"
    if "distribution" in lowered:
        return "distribution regime"
    if "sideways" in lowered:
        return "sideways low-conviction regime"
    if "uptrend" in lowered and "high" in lowered:
        return "uptrend with higher volatility"
    if "uptrend" in lowered and "low" in lowered:
        return "bullish low-volatility trend"
    return f"{text} regime"


def _relative_strength_phrase(rank: object) -> str | None:
    value = _clean_float(rank)
    if value is None:
        return None
    if value.is_integer():
        rank_text = f"#{int(value)}"
    else:
        rank_text = f"#{value:.1f}"
    return f"relative strength rank {rank_text}"


def _action_position_phrase(action: object, target_bucket: object) -> str:
    action_text = str(action or "Watch").strip() or "Watch"
    bucket = str(target_bucket or "0%").strip() or "0%"
    if action_text == "Add":
        return f"Add toward {bucket} of max position"
    if action_text == "Hold":
        return f"Hold near {bucket} of max position"
    if action_text == "Trim":
        return f"Trim toward {bucket} of max position"
    if action_text == "Exit":
        return f"Exit; target {bucket} exposure"
    if action_text == "Watch":
        return f"Watch; target {bucket} exposure"
    return f"{action_text}; target {bucket} exposure"


def one_line_reason(row: pd.Series) -> str:
    """Generate a concise reason for the decision table."""

    action = row.get("Suggested action", row.get("Suggested Action", "Watch"))
    target_bucket = row.get("Target exposure bucket", row.get("Target Exposure Bucket", "0%"))
    regime = str(row.get("Rule-based regime", row.get("Rule-Based Regime", "")))
    score = row.get("ML score", row.get("ML Score", pd.NA))
    risk = row.get("Drawdown-risk probability", row.get("ML Drawdown-Risk Probability", pd.NA))
    rs_rank = row.get("Relative strength rank", pd.NA)

    parts = [
        _action_position_phrase(action, target_bucket),
        _regime_phrase(regime),
        _score_phrase(score),
        _risk_phrase(risk),
    ]
    rs_phrase = _relative_strength_phrase(rs_rank)
    if rs_phrase:
        parts.append(rs_phrase)
    return "; ".join(parts) + "."


def build_decision_table(
    current_scores: pd.DataFrame,
    latest_features: pd.DataFrame,
    config: DecisionConfig,
    profile: DecisionProfile,
    current_weights: pd.Series | None = None,
    benchmark: str = "SPY",
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

    benchmark_column = f"rs_{benchmark.lower()}_60d"
    if benchmark_column in allocated:
        rs_column = benchmark_column
    elif "rs_spy_60d" in allocated:
        rs_column = "rs_spy_60d"
    else:
        rs_column = "RS vs SPY 60d"
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
    output["_action_order"] = output["Suggested action"].map(ACTION_RANK).fillna(len(ACTION_RANK))
    return output.sort_values(["_action_order", "ML score"], ascending=[True, False]).drop(
        columns=["_action_order"]
    ).reset_index(drop=True)


def action_counts(decision_table: pd.DataFrame) -> dict[str, int]:
    """Count Add/Hold/Trim/Exit/Watch labels."""

    counts = decision_table.get("Suggested action", pd.Series(dtype=str)).value_counts().to_dict()
    return {action: int(counts.get(action, 0)) for action in ACTION_ORDER}
