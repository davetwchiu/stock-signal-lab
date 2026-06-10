"""Persistent user portfolio stock list for Decision Mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml

from src.utils.config import PROJECT_ROOT


USER_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "user_portfolio.yaml"
T = TypeVar("T")


@dataclass(frozen=True)
class UserPortfolio:
    """A saved local portfolio list."""

    name: str
    tickers: tuple[str, ...]
    updated_at: str


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
