"""Locked default configuration for Decision Mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import PROJECT_ROOT


DEFAULT_DECISION_CONFIG_PATH = PROJECT_ROOT / "config" / "default_decision_mode.yaml"
DEFAULT_ADVANCED_OVERRIDE = False


@dataclass(frozen=True)
class DecisionProfile:
    """Small user-facing profile override."""

    cash_floor: float
    max_single_position_exposure: float
    high_drawdown_risk_threshold: float
    moderate_drawdown_risk_threshold: float


@dataclass(frozen=True)
class DecisionConfig:
    """Decision Mode defaults loaded from YAML."""

    default_ticker_universe: tuple[str, ...]
    default_benchmark: str
    default_label_horizon: int
    default_model: str
    default_feature_group: str
    ml_score_thresholds: dict[str, float]
    drawdown_risk_thresholds: dict[str, float]
    allocation_buckets: tuple[int, ...]
    default_rebalance_frequency: str
    default_transaction_cost_bps: float
    default_slippage_bps: float
    default_cash_floor: float
    default_max_single_position_exposure: float
    profiles: dict[str, DecisionProfile]


def load_decision_config(path: Path = DEFAULT_DECISION_CONFIG_PATH) -> DecisionConfig:
    """Load the locked Decision Mode YAML config."""

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = {
        name: DecisionProfile(
            cash_floor=float(values["cash_floor"]),
            max_single_position_exposure=float(values["max_single_position_exposure"]),
            high_drawdown_risk_threshold=float(values["high_drawdown_risk_threshold"]),
            moderate_drawdown_risk_threshold=float(values["moderate_drawdown_risk_threshold"]),
        )
        for name, values in raw["profiles"].items()
    }
    return DecisionConfig(
        default_ticker_universe=tuple(raw["default_ticker_universe"]),
        default_benchmark=str(raw["default_benchmark"]),
        default_label_horizon=int(raw["default_label_horizon"]),
        default_model=str(raw["default_model"]),
        default_feature_group=str(raw["default_feature_group"]),
        ml_score_thresholds={key: float(value) for key, value in raw["ml_score_thresholds"].items()},
        drawdown_risk_thresholds={key: float(value) for key, value in raw["drawdown_risk_thresholds"].items()},
        allocation_buckets=tuple(int(value) for value in raw["allocation_buckets"]),
        default_rebalance_frequency=str(raw["default_rebalance_frequency"]),
        default_transaction_cost_bps=float(raw["default_transaction_cost_bps"]),
        default_slippage_bps=float(raw["default_slippage_bps"]),
        default_cash_floor=float(raw["default_cash_floor"]),
        default_max_single_position_exposure=float(raw["default_max_single_position_exposure"]),
        profiles=profiles,
    )


def profile_settings(config: DecisionConfig, profile_name: str) -> DecisionProfile:
    """Return a named preset profile, falling back to Balanced."""

    return config.profiles.get(profile_name, config.profiles["Balanced"])
