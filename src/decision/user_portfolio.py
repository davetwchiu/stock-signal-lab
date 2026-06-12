"""Persistent user portfolio stock lists for Decision Mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml

from src.utils.config import PROJECT_ROOT


USER_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "user_portfolio.yaml"
USER_PORTFOLIOS_PATH = PROJECT_ROOT / "data" / "user_portfolios.yaml"
DEFAULT_PORTFOLIO_LIST_NAME = "Main"
T = TypeVar("T")


@dataclass(frozen=True)
class UserPortfolio:
    """A saved local portfolio list."""

    name: str
    tickers: tuple[str, ...]
    updated_at: str


@dataclass(frozen=True)
class UserPortfolioLists:
    """Saved local named portfolio lists."""

    active_name: str
    portfolios: tuple[UserPortfolio, ...]
    source: str

    @property
    def names(self) -> tuple[str, ...]:
        """Return list names in saved order."""

        return tuple(portfolio.name for portfolio in self.portfolios)

    @property
    def active(self) -> UserPortfolio:
        """Return the active portfolio list."""

        for portfolio in self.portfolios:
            if portfolio.name == self.active_name:
                return portfolio
        return self.portfolios[0]


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols while preserving hyphenated symbols such as BRK-B."""

    return ticker.strip().upper().replace("/", "-").replace(" ", "")


def parse_portfolio_tickers(raw: str) -> list[str]:
    """Parse comma-separated or newline-separated tickers."""

    normalized = raw.replace("\n", ",").replace("\t", ",")
    seen: set[str] = set()
    output: list[str] = []
    for part in normalized.split(","):
        ticker = normalize_ticker(part)
        if ticker and ticker not in seen:
            seen.add(ticker)
            output.append(ticker)
    return output


def load_user_portfolio(path: Path = USER_PORTFOLIO_PATH) -> UserPortfolio | None:
    """Load a saved portfolio list, returning None when missing or invalid."""

    if not path.exists():
        return None
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    portfolio = raw.get("portfolio", {})
    tickers = parse_portfolio_tickers("\n".join(str(ticker) for ticker in portfolio.get("tickers", [])))
    if not tickers:
        return None
    return UserPortfolio(
        name=str(portfolio.get("name", "My Portfolio")),
        tickers=tuple(tickers),
        updated_at=str(raw.get("updated_at", "")),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_list_name(name: str) -> str:
    return name.strip()


def _portfolio_by_name(collection: UserPortfolioLists, name: str) -> UserPortfolio | None:
    for portfolio in collection.portfolios:
        if portfolio.name == name:
            return portfolio
    return None


def _collection_with(
    portfolios: tuple[UserPortfolio, ...],
    active_name: str,
    source: str = "multi",
) -> UserPortfolioLists:
    if not portfolios:
        raise ValueError("At least one portfolio list is required.")
    if all(portfolio.name != active_name for portfolio in portfolios):
        active_name = portfolios[0].name
    return UserPortfolioLists(active_name=active_name, portfolios=portfolios, source=source)


def default_user_portfolio_lists(
    system_default_tickers: tuple[str, ...] | list[str],
    name: str = DEFAULT_PORTFOLIO_LIST_NAME,
) -> UserPortfolioLists:
    """Return an unsaved default portfolio-list collection."""

    tickers = parse_portfolio_tickers("\n".join(system_default_tickers))
    return UserPortfolioLists(
        active_name=name,
        portfolios=(UserPortfolio(name=name, tickers=tuple(tickers), updated_at=""),),
        source="default",
    )


def load_user_portfolio_lists(
    path: Path = USER_PORTFOLIOS_PATH,
    legacy_path: Path = USER_PORTFOLIO_PATH,
) -> UserPortfolioLists | None:
    """Load named portfolio lists, falling back to the legacy single-list file."""

    if path.exists():
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_portfolios = raw.get("portfolios", {})
        portfolios: list[UserPortfolio] = []
        if isinstance(raw_portfolios, dict):
            for raw_name, raw_portfolio in raw_portfolios.items():
                if not isinstance(raw_portfolio, dict):
                    continue
                name = _clean_list_name(str(raw_name))
                tickers = parse_portfolio_tickers(
                    "\n".join(str(ticker) for ticker in raw_portfolio.get("tickers", []))
                )
                if name and tickers:
                    portfolios.append(
                        UserPortfolio(
                            name=name,
                            tickers=tuple(tickers),
                            updated_at=str(raw_portfolio.get("updated_at", "")),
                        )
                    )
        if portfolios:
            active_name = _clean_list_name(str(raw.get("active", ""))) or portfolios[0].name
            return _collection_with(tuple(portfolios), active_name=active_name, source="multi")
        return None

    legacy = load_user_portfolio(legacy_path)
    if legacy is None:
        return None
    main = UserPortfolio(
        name=DEFAULT_PORTFOLIO_LIST_NAME,
        tickers=legacy.tickers,
        updated_at=legacy.updated_at,
    )
    return UserPortfolioLists(
        active_name=DEFAULT_PORTFOLIO_LIST_NAME,
        portfolios=(main,),
        source="legacy",
    )


def save_user_portfolio_lists(
    collection: UserPortfolioLists,
    path: Path = USER_PORTFOLIOS_PATH,
) -> Path:
    """Persist named portfolio lists to YAML."""

    if not collection.portfolios:
        raise ValueError("At least one portfolio list is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active": collection.active.name,
        "portfolios": {
            portfolio.name: {
                "tickers": list(portfolio.tickers),
                "updated_at": portfolio.updated_at or _now_iso(),
            }
            for portfolio in collection.portfolios
        },
        "updated_at": _now_iso(),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def set_active_user_portfolio_list(collection: UserPortfolioLists, name: str) -> UserPortfolioLists:
    """Return a collection with a new active list."""

    clean_name = _clean_list_name(name)
    if _portfolio_by_name(collection, clean_name) is None:
        raise ValueError(f"Unknown portfolio list: {name}")
    return UserPortfolioLists(active_name=clean_name, portfolios=collection.portfolios, source="multi")


def save_user_portfolio_list(
    collection: UserPortfolioLists,
    name: str,
    tickers: list[str],
) -> UserPortfolioLists:
    """Return a collection with one named list updated."""

    clean_name = _clean_list_name(name)
    clean_tickers = parse_portfolio_tickers("\n".join(tickers))
    if not clean_name:
        raise ValueError("Portfolio list name is required.")
    if not clean_tickers:
        raise ValueError("Cannot save an empty portfolio stock list.")

    updated = UserPortfolio(name=clean_name, tickers=tuple(clean_tickers), updated_at=_now_iso())
    portfolios = tuple(
        updated if portfolio.name == clean_name else portfolio
        for portfolio in collection.portfolios
    )
    if _portfolio_by_name(collection, clean_name) is None:
        portfolios = collection.portfolios + (updated,)
    return _collection_with(portfolios, active_name=clean_name)


def create_user_portfolio_list(
    collection: UserPortfolioLists,
    name: str,
    tickers: list[str],
) -> UserPortfolioLists:
    """Return a collection with a newly named list."""

    clean_name = _clean_list_name(name)
    if not clean_name:
        raise ValueError("Portfolio list name is required.")
    if _portfolio_by_name(collection, clean_name) is not None:
        raise ValueError(f"Portfolio list already exists: {clean_name}")
    return save_user_portfolio_list(collection, clean_name, tickers)


def delete_user_portfolio_list(collection: UserPortfolioLists, name: str) -> UserPortfolioLists:
    """Return a collection with one named list removed."""

    clean_name = _clean_list_name(name)
    if len(collection.portfolios) <= 1:
        raise ValueError("Cannot delete the last portfolio list.")
    remaining = tuple(portfolio for portfolio in collection.portfolios if portfolio.name != clean_name)
    if len(remaining) == len(collection.portfolios):
        raise ValueError(f"Unknown portfolio list: {name}")
    active_name = collection.active_name if collection.active_name != clean_name else remaining[0].name
    return _collection_with(remaining, active_name=active_name)


def resolve_user_portfolio_lists(
    system_default_tickers: tuple[str, ...] | list[str],
    path: Path = USER_PORTFOLIOS_PATH,
    legacy_path: Path = USER_PORTFOLIO_PATH,
) -> UserPortfolioLists:
    """Return saved named lists, legacy list, or system defaults."""

    saved = load_user_portfolio_lists(path=path, legacy_path=legacy_path)
    if saved is not None:
        return saved
    return default_user_portfolio_lists(system_default_tickers)


def save_user_portfolio(
    tickers: list[str],
    path: Path = USER_PORTFOLIO_PATH,
    name: str = "My Portfolio",
) -> Path:
    """Persist the user portfolio list to YAML."""

    clean_tickers = parse_portfolio_tickers("\n".join(tickers))
    if not clean_tickers:
        raise ValueError("Cannot save an empty portfolio stock list.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "portfolio": {
            "name": name,
            "tickers": clean_tickers,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def resolve_active_portfolio_tickers(
    system_default_tickers: tuple[str, ...] | list[str],
    path: Path = USER_PORTFOLIO_PATH,
) -> tuple[list[str], UserPortfolio | None]:
    """Return saved portfolio tickers when present, otherwise system defaults."""

    saved = load_user_portfolio(path)
    if saved is not None:
        return list(saved.tickers), saved
    return parse_portfolio_tickers("\n".join(system_default_tickers)), None


def select_active_portfolio_frames(feature_frames: dict[str, T], active_tickers: list[str]) -> dict[str, T]:
    """Filter feature frames to the exact active portfolio order used for scoring."""

    return {ticker: feature_frames[ticker] for ticker in active_tickers if ticker in feature_frames}
