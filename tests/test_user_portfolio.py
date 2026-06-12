from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.decision.config import load_decision_config
from src.decision.user_portfolio import (
    UserPortfolio,
    UserPortfolioLists,
    create_user_portfolio_list,
    delete_user_portfolio_list,
    load_user_portfolio,
    load_user_portfolio_lists,
    parse_portfolio_tickers,
    resolve_active_portfolio_tickers,
    resolve_user_portfolio_lists,
    save_user_portfolio,
    save_user_portfolio_list,
    save_user_portfolio_lists,
    select_active_portfolio_frames,
    set_active_user_portfolio_list,
)


EXPECTED_DEFAULT_PORTFOLIO = [
    "AAPL",
    "AMAT",
    "ANET",
    "AVGO",
    "BRK-B",
    "ETN",
    "GOOG",
    "GRID",
    "KTOS",
    "MRVL",
    "NVDA",
    "ONDS",
    "PLTR",
    "RDW",
    "RKLB",
    "TSLA",
    "TSM",
    "UUUU",
    "XLK",
    "SEMI.AS",
    "200A.T",
    "2854.T",
    "3587.TWO",
    "00935.TW",
    "ABBN.SW",
    "IART.SW",
    "SMSD.IL",
    "WTAI.L",
]


def test_parse_comma_separated_tickers() -> None:
    assert parse_portfolio_tickers("aapl, msft, nvda") == ["AAPL", "MSFT", "NVDA"]


def test_parse_newline_separated_tickers() -> None:
    assert parse_portfolio_tickers("aapl\nmsft\nnvda") == ["AAPL", "MSFT", "NVDA"]


def test_parse_removes_duplicates_and_preserves_order() -> None:
    assert parse_portfolio_tickers("nvda, AMD, nvda, amd, TSM") == ["NVDA", "AMD", "TSM"]


def test_parse_uppercases_and_preserves_hyphen_ticker() -> None:
    assert parse_portfolio_tickers("brk-b, goog") == ["BRK-B", "GOOG"]


def test_parse_preserves_exchange_suffixes_and_uses_yahoo_style_semiconductor_etf() -> None:
    assert parse_portfolio_tickers("semi.as, 3587.two, 00935.tw") == ["SEMI.AS", "3587.TWO", "00935.TW"]


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


def test_legacy_single_list_file_loads_as_main_named_list(tmp_path: Path) -> None:
    legacy_path = tmp_path / "user_portfolio.yaml"
    multi_path = tmp_path / "user_portfolios.yaml"
    save_user_portfolio(["semi.as", "3587.two"], path=legacy_path)

    loaded = load_user_portfolio_lists(path=multi_path, legacy_path=legacy_path)

    assert loaded is not None
    assert loaded.source == "legacy"
    assert loaded.active_name == "Main"
    assert loaded.names == ("Main",)
    assert loaded.active.tickers == ("SEMI.AS", "3587.TWO")
    assert "SEMI.AEB" not in loaded.active.tickers


def test_save_and_load_multiple_named_portfolio_lists(tmp_path: Path) -> None:
    path = tmp_path / "user_portfolios.yaml"
    collection = UserPortfolioLists(
        active_name="Income",
        portfolios=(
            UserPortfolio(name="Core", tickers=("AAPL", "MSFT"), updated_at="2026-01-01T00:00:00+00:00"),
            UserPortfolio(name="Income", tickers=("BRK-B", "SEMI.AS"), updated_at="2026-01-02T00:00:00+00:00"),
        ),
        source="multi",
    )

    save_user_portfolio_lists(collection, path=path)
    loaded = load_user_portfolio_lists(path=path, legacy_path=tmp_path / "missing_legacy.yaml")

    assert loaded is not None
    assert loaded.source == "multi"
    assert loaded.names == ("Core", "Income")
    assert loaded.active_name == "Income"
    assert loaded.active.tickers == ("BRK-B", "SEMI.AS")


def test_active_portfolio_list_persists_after_save(tmp_path: Path) -> None:
    path = tmp_path / "user_portfolios.yaml"
    collection = UserPortfolioLists(
        active_name="Core",
        portfolios=(
            UserPortfolio(name="Core", tickers=("AAPL",), updated_at=""),
            UserPortfolio(name="Growth", tickers=("NVDA",), updated_at=""),
        ),
        source="multi",
    )

    collection = set_active_user_portfolio_list(collection, "Growth")
    save_user_portfolio_lists(collection, path=path)
    loaded = load_user_portfolio_lists(path=path, legacy_path=tmp_path / "missing_legacy.yaml")

    assert loaded is not None
    assert loaded.active_name == "Growth"
    assert loaded.active.tickers == ("NVDA",)


def test_create_new_portfolio_list_uses_current_tickers() -> None:
    collection = UserPortfolioLists(
        active_name="Main",
        portfolios=(UserPortfolio(name="Main", tickers=("AAPL",), updated_at=""),),
        source="default",
    )

    created = create_user_portfolio_list(collection, "International", ["semi.as", "00935.tw"])

    assert created.active_name == "International"
    assert created.names == ("Main", "International")
    assert created.active.tickers == ("SEMI.AS", "00935.TW")
    assert "SEMI.AEB" not in created.active.tickers


def test_update_selected_portfolio_list_tickers_preserves_exact_symbols() -> None:
    collection = UserPortfolioLists(
        active_name="Main",
        portfolios=(UserPortfolio(name="Main", tickers=("AAPL",), updated_at=""),),
        source="multi",
    )

    updated = save_user_portfolio_list(collection, "Main", ["semi.as", "3587.two", "00935.tw"])

    assert updated.active_name == "Main"
    assert updated.active.tickers == ("SEMI.AS", "3587.TWO", "00935.TW")
    assert "SEMI.AEB" not in updated.active.tickers


def test_delete_portfolio_list_preserves_at_least_one_list() -> None:
    collection = UserPortfolioLists(
        active_name="Growth",
        portfolios=(
            UserPortfolio(name="Core", tickers=("AAPL",), updated_at=""),
            UserPortfolio(name="Growth", tickers=("NVDA",), updated_at=""),
        ),
        source="multi",
    )

    deleted = delete_user_portfolio_list(collection, "Growth")

    assert deleted.names == ("Core",)
    assert deleted.active_name == "Core"
    try:
        delete_user_portfolio_list(deleted, "Core")
    except ValueError as error:
        assert "last portfolio list" in str(error)
    else:
        raise AssertionError("Deleting the last portfolio list should fail.")


def test_resolve_falls_back_to_system_default_when_no_saved_file(tmp_path: Path) -> None:
    config = load_decision_config()

    tickers, saved = resolve_active_portfolio_tickers(config.default_ticker_universe, path=tmp_path / "missing.yaml")

    assert saved is None
    assert tickers == EXPECTED_DEFAULT_PORTFOLIO
    assert "SEMI.AS" in tickers
    assert "SEMI.AEB" not in tickers


def test_resolve_named_lists_falls_back_to_system_default_when_no_saved_file(tmp_path: Path) -> None:
    config = load_decision_config()

    collection = resolve_user_portfolio_lists(
        config.default_ticker_universe,
        path=tmp_path / "missing_multi.yaml",
        legacy_path=tmp_path / "missing_legacy.yaml",
    )

    assert collection.source == "default"
    assert collection.active_name == "Main"
    assert list(collection.active.tickers) == EXPECTED_DEFAULT_PORTFOLIO
    assert "SEMI.AS" in collection.active.tickers
    assert "SEMI.AEB" not in collection.active.tickers


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
