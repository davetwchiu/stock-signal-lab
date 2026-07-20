from __future__ import annotations

import pandas as pd

from src.decision.broad_pullback import SECTOR_PULLBACK_TICKERS, build_broad_pullback_playbook


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Adj Close": values},
        index=pd.date_range("2025-01-02", periods=len(values), freq="B"),
    )


def _sector_prices(daily_return: float = -0.01) -> list[float]:
    prices = [100.0 + index * 0.1 for index in range(215)]
    for _ in range(5):
        prices.append(prices[-1] * (1.0 + daily_return))
    return prices


def _frames() -> dict[str, pd.DataFrame]:
    frames = {"SPY": _frame([100.0 + index * 0.2 for index in range(220)])}
    frames.update({ticker: _frame(_sector_prices()) for ticker in SECTOR_PULLBACK_TICKERS})
    return frames


def test_exact_nine_sector_pullback_confirms_hold_playbook() -> None:
    playbook = build_broad_pullback_playbook(_frames())

    assert playbook.status == "Confirmed"
    assert playbook.action == "Hold / review position size"
    assert playbook.negative_sector_count == 9
    assert playbook.benchmark_trend == "above 200-day average"
    assert "72.9%" in playbook.evidence
    assert "not an automatic buy signal" in playbook.evidence


def test_one_positive_sector_withholds_historical_statistics() -> None:
    frames = _frames()
    frames["XLY"] = _frame(_sector_prices(0.01))

    playbook = build_broad_pullback_playbook(frames)

    assert playbook.status == "Not confirmed"
    assert playbook.negative_sector_count == 8
    assert "72.9%" not in playbook.evidence
    assert "statistics do not apply" in playbook.evidence


def test_spy_below_200dma_withholds_pullback_playbook() -> None:
    frames = _frames()
    frames["SPY"] = _frame([100.0] * 214 + [90.0] * 6)

    playbook = build_broad_pullback_playbook(frames)

    assert playbook.status == "Not confirmed"
    assert playbook.negative_sector_count == 9
    assert playbook.benchmark_trend == "below 200-day average"
    assert "72.9%" not in playbook.evidence


def test_missing_sector_prices_make_exact_rule_unavailable() -> None:
    frames = _frames()
    del frames["XLU"]

    playbook = build_broad_pullback_playbook(frames)

    assert playbook.status == "Unavailable"
    assert playbook.available_sector_count == 8
    assert "8/9" in playbook.evidence
    assert "72.9%" not in playbook.evidence
