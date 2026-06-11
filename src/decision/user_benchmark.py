"""Persistent user benchmark for Decision Mode."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import PROJECT_ROOT


USER_BENCHMARK_PATH = PROJECT_ROOT / "data" / "user_benchmark.yaml"


@dataclass(frozen=True)
class UserBenchmark:
    """A saved local benchmark selection."""

    benchmark: str
    updated_at: str


def normalize_benchmark(benchmark: str) -> str:
    """Normalize a benchmark ticker."""

    return benchmark.strip().upper().replace("/", "-").replace(" ", "")


def _allowed_benchmarks(allowed_benchmarks: Collection[str] | None) -> set[str]:
    if allowed_benchmarks is None:
        return set()
    return {normalize_benchmark(benchmark) for benchmark in allowed_benchmarks}


def load_user_benchmark(
    path: Path = USER_BENCHMARK_PATH,
    allowed_benchmarks: Collection[str] | None = None,
) -> UserBenchmark | None:
    """Load a saved benchmark, returning None when missing or invalid."""

    if not path.exists():
        return None
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    benchmark = normalize_benchmark(str(raw.get("benchmark", "")))
    allowed = _allowed_benchmarks(allowed_benchmarks)
    if not benchmark or (allowed and benchmark not in allowed):
        return None
    return UserBenchmark(
        benchmark=benchmark,
        updated_at=str(raw.get("updated_at", "")),
    )


def save_user_benchmark(
    benchmark: str,
    path: Path = USER_BENCHMARK_PATH,
) -> Path:
    """Persist the user benchmark selection to YAML."""

    clean_benchmark = normalize_benchmark(benchmark)
    if not clean_benchmark:
        raise ValueError("Cannot save an empty benchmark.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": clean_benchmark,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def resolve_active_benchmark(
    system_default_benchmark: str,
    path: Path = USER_BENCHMARK_PATH,
    allowed_benchmarks: Collection[str] | None = None,
) -> str:
    """Return the saved benchmark when present, otherwise the system default."""

    saved = load_user_benchmark(path=path, allowed_benchmarks=allowed_benchmarks)
    if saved is not None:
        return saved.benchmark
    return normalize_benchmark(system_default_benchmark)
