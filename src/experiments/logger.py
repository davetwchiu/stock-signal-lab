"""Local JSON experiment logging."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.experiments.schema import ExperimentRecord
from src.utils.config import DATA_DIR


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def write_experiment_log(
    record: ExperimentRecord,
    experiment_dir: Path | None = None,
) -> Path:
    """Write one experiment record as a local JSON file."""

    directory = experiment_dir or (DATA_DIR / "experiments")
    directory.mkdir(parents=True, exist_ok=True)
    name = _safe_name(f"{record.timestamp}_{record.feature_group}_{record.model_type}") + ".json"
    path = directory / name
    path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
    return path


def append_experiment_index(path: Path, index_path: Path | None = None) -> Path:
    """Append a simple CSV index entry for easier browsing."""

    target = index_path or path.parent / "index.csv"
    row = pd.DataFrame([{"path": str(path), "filename": path.name}])
    if target.exists():
        existing = pd.read_csv(target)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(target, index=False)
    return target

