"""Experiment log schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    """Serializable metadata for one research run."""

    ticker_universe: list[str]
    date_range: tuple[str, str]
    feature_group: str
    model_type: str
    label_type: str
    threshold_settings: dict[str, Any]
    transaction_cost_assumptions: dict[str, Any]
    validation_settings: dict[str, Any]
    portfolio_settings: dict[str, Any]
    key_metrics: dict[str, Any]
    output_file_paths: dict[str, str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def default_experiment_dir(project_root: Path) -> Path:
    """Return the local experiment output directory."""

    return project_root / "data" / "experiments"

