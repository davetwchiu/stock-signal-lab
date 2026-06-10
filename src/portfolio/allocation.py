"""Transparent allocation rules for Stage 3 portfolio research."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.features.regime import DISTRIBUTION, DOWNTREND_HIGH_RISK, UPTREND_HIGH_VOL, UPTREND_LOW_VOL


@dataclass(frozen=True)
class AllocationConfig:
    """Configurable guardrails for target-weight generation."""

    max_position_size: float = 0.12
    min_position_size: float = 0.01
    cash_floor: float = 0.10
    max_gross_exposure: float = 0.90
    drawdown_risk_threshold: float = 0.60
    moderate_drawdown_risk_threshold: float = 0.40
    volatility_target: float | None = None
    score_band_multipliers: tuple[tuple[float, float, float], ...] = (
        (80.0, 100.0, 1.00),
        (60.0, 80.0, 0.75),
        (40.0, 60.0, 0.50),
        (20.0, 40.0, 0.25),
        (0.0, 20.0, 0.00),
    )
    regime_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            UPTREND_LOW_VOL: 1.00,
            UPTREND_HIGH_VOL: 0.85,
            "Sideways / low conviction": 0.50,
            DISTRIBUTION: 0.25,
            DOWNTREND_HIGH_RISK: 0.00,
        }
    )


def score_multiplier(score: float, config: AllocationConfig) -> float:
    """Map an ML score to an allowed-position multiplier."""

    if pd.isna(score):
        return 0.0
    for low, high, multiplier in config.score_band_multipliers:
        if low <= float(score) <= high:
            return multiplier
    return 0.0


def suggested_action(current_weight: float, target_weight: float, min_position_size: float = 0.01) -> str:
    """Convert current and target weights into a decision-support action."""

    if target_weight <= 0 and current_weight > min_position_size:
        return "Exit"
    if target_weight <= 0:
        return "Watch"
    if current_weight <= min_position_size and target_weight > min_position_size:
        return "Add"
    if target_weight > current_weight + min_position_size:
        return "Add"
    if target_weight < current_weight - min_position_size:
        return "Trim"
    return "Hold"


def allocate_from_scores(
    scores: pd.DataFrame,
    config: AllocationConfig | None = None,
    current_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Create target weights from ML score, drawdown risk, regime, rank, and volatility."""

    cfg = config or AllocationConfig()
    if scores.empty:
        return pd.DataFrame(columns=["Ticker", "target_weight", "suggested_action", "allocation_explanation"])

    frame = scores.copy()
    if "Ticker" not in frame:
        raise ValueError("scores must include a Ticker column")
    active_weights = current_weights if current_weights is not None else pd.Series(dtype=float)
    frame["current_weight"] = frame["Ticker"].map(active_weights.to_dict()).fillna(0.0)
    targets: list[float] = []
    explanations: list[str] = []

    for _, row in frame.iterrows():
        ticker = row["Ticker"]
        score = float(row.get("ML Score", row.get("ml_score", 0.0)) or 0.0)
        risk_probability = float(
            row.get("ML Drawdown-Risk Probability", row.get("drawdown_risk_probability", 1.0)) or 1.0
        )
        regime = str(row.get("Rule-Based Regime", row.get("regime", "")))
        volatility = row.get("volatility_60d", pd.NA)

        multiplier = score_multiplier(score, cfg)
        reasons = [f"score {score:.0f}"]

        if risk_probability >= cfg.drawdown_risk_threshold:
            multiplier = 0.0
            reasons.append("high drawdown risk")
        elif risk_probability >= cfg.moderate_drawdown_risk_threshold:
            multiplier *= 0.5
            reasons.append("moderate drawdown risk")

        regime_multiplier = cfg.regime_multipliers.get(regime, 0.5)
        multiplier *= regime_multiplier
        reasons.append(f"regime multiplier {regime_multiplier:.2f}")

        if cfg.volatility_target and pd.notna(volatility) and float(volatility) > 0:
            vol_multiplier = min(1.0, cfg.volatility_target / float(volatility))
            multiplier *= vol_multiplier
            reasons.append(f"volatility adjustment {vol_multiplier:.2f}")

        target = cfg.max_position_size * multiplier
        if 0 < target < cfg.min_position_size:
            target = 0.0
        targets.append(target)
        explanations.append(f"{ticker}: " + "; ".join(reasons))

    frame["raw_target_weight"] = targets
    exposure_cap = min(cfg.max_gross_exposure, 1.0 - cfg.cash_floor)
    gross = float(frame["raw_target_weight"].sum())
    scale = exposure_cap / gross if gross > exposure_cap and gross > 0 else 1.0
    frame["target_weight"] = frame["raw_target_weight"] * scale
    frame["suggested_action"] = [
        suggested_action(current, target, cfg.min_position_size)
        for current, target in zip(frame["current_weight"], frame["target_weight"], strict=False)
    ]
    frame["allocation_explanation"] = explanations
    if scale < 1.0:
        frame["allocation_explanation"] += f"; scaled to exposure cap {exposure_cap:.0%}"
    return frame
