from __future__ import annotations

from pathlib import Path

import yaml

from src.decision.user_benchmark import load_user_benchmark, resolve_active_benchmark, save_user_benchmark


def test_save_benchmark_writes_yaml(tmp_path: Path) -> None:
    path = tmp_path / "user_benchmark.yaml"

    save_user_benchmark("qqq", path=path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["benchmark"] == "QQQ"
    assert raw["updated_at"]


def test_load_user_benchmark_uses_saved_file(tmp_path: Path) -> None:
    path = tmp_path / "user_benchmark.yaml"
    save_user_benchmark("smh", path=path)

    loaded = load_user_benchmark(path)

    assert loaded is not None
    assert loaded.benchmark == "SMH"


def test_resolve_benchmark_falls_back_to_system_default_when_no_saved_file(tmp_path: Path) -> None:
    benchmark = resolve_active_benchmark("SPY", path=tmp_path / "missing.yaml", allowed_benchmarks=["SPY", "QQQ"])

    assert benchmark == "SPY"


def test_resolve_benchmark_uses_saved_benchmark_when_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "user_benchmark.yaml"
    save_user_benchmark("qqq", path=path)

    benchmark = resolve_active_benchmark("SPY", path=path, allowed_benchmarks=["SPY", "QQQ"])

    assert benchmark == "QQQ"


def test_resolve_benchmark_falls_back_when_saved_benchmark_is_not_allowed(tmp_path: Path) -> None:
    path = tmp_path / "user_benchmark.yaml"
    save_user_benchmark("iwm", path=path)

    benchmark = resolve_active_benchmark("SPY", path=path, allowed_benchmarks=["SPY", "QQQ"])

    assert benchmark == "SPY"
