from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.decision.config import load_decision_config
from src.decision.user_portfolio import (
    load_user_portfolio,
    parse_portfolio_tickers,
    resolve_active_portfolio_tickers,
    save_user_portfolio,
    select_active_portfolio_frames,
)


def test_parse_comma_separated_tickers() -> None:
    assert parse_portfolio_tickers("aapl, msft, nvda") == ["AAPL", "MSFT", "NVDA"]


def test_parse_newline_separated_tickers() -> None:
    assert parse_portfolio_tickers("aapl\nmsft\nnvda") == ["AAPL", "MSFT", "NVDA"]


def test_parse_removes_duplicates_and_preserves_order() -> None:
    assert parse_portfolio_tickers("nvda, AMD, nvda, amd, TSM") == ["NVDA", "AMD", "TSM"]


def test_parse_uppercases_and_preserves_hyphen_ticker() -> None:
    assert parse_portfolio_tickers("brk-b, goog") == ["BRK-B", "GOOG"]


def test_save_portfolio_list_writes_yaml(tmp_path: Path) -> None:
    path = tmp_path / "user_portfolio.yaml"

    save_user_portfolio(["aapl", "AMD", "BRK-B"], path=path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["portfolio"]["name"] == "My Portfolio"
    assert raw["portfolio"]["tickers"] == ["AAPL", "AMD", "BRK-B"]
    assert raw["updated_at"]


def test_load_user_portfolio_uses_saved_file(tmp_path: Path) -> None:
    path = tmp_path / "user_portfolio.yaml"
    save_user_portfolio(["aapl", "msft"], path=path)

    loaded = load_user_portfolio(path)

    assert loaded is not None
    assert loaded.tickers == ("AAPL", "MSFT")


def test_resolve_falls_back_to_system_default_when_no_saved_file(tmp_path: Path) -> None:
    config = load_decision_config()

    tickers, saved = resolve_active_portfolio_tickers(config.default_ticker_universe, path=tmp_path / "missing.yaml")

    assert saved is None
    assert tickers == list(config.default_ticker_universe)


def test_resolve_uses_saved_portfolio_when_file_exists(tmp_path: Path) -> None:
    config = load_decision_config()
    path = tmp_path / "user_portfolio.yaml"
    save_user_portfolio(["aapl", "msft"], path=path)

    tickers, saved = resolve_active_portfolio_tickers(config.default_ticker_universe, path=path)

    assert saved is not None
    assert tickers == ["AAPL", "MSFT"]


def test_advanced_override_off_does_not_reset_saved_portfolio(tmp_path: Path) -> None:
    config = load_decision_config()
    path = tmp_path / "user_portfolio.yaml"
    save_user_portfolio(["aapl", "msft"], path=path)

    advanced_override = False
    tickers, _ = resolve_active_portfolio_tickers(config.default_ticker_universe, path=path)

    assert advanced_override is False
    assert tickers == ["AAPL", "MSFT"]


def test_scoring_frames_match_active_saved_portfolio_list() -> None:
    active_tickers = ["AAPL", "MSFT"]
    feature_frames = {
        "AAPL": pd.DataFrame({"x": [1]}),
        "MSFT": pd.DataFrame({"x": [2]}),
        "SPY": pd.DataFrame({"x": [3]}),
        "QQQ": pd.DataFrame({"x": [4]}),
    }

    scoring_frames = select_active_portfolio_frames(feature_frames, active_tickers)

    assert list(scoring_frames) == active_tickers
