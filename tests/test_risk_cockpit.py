from __future__ import annotations

import pandas as pd

from src.decision.risk_cockpit import (
    MARKET_STATES,
    SINGLE_NAME_STATES,
    THEME_STATES,
    build_market_stress_panel,
    build_risk_cockpit,
    build_theme_stress_panel,
    classify_single_name_health,
)


def frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Adj Close": values},
        index=pd.date_range("2025-01-01", periods=len(values), freq="D"),
    )


def ramp(start: float, end: float, rows: int = 261) -> list[float]:
    return list(pd.Series(range(rows)).map(lambda idx: start + (end - start) * idx / (rows - 1)))


def test_market_stress_flags_qqq_breakdown_against_spy() -> None:
    frames = {
        "SPY": frame([100.0] * 261),
        "QQQ": frame([120.0] * 230 + [90.0] * 31),
        "TLT": frame(ramp(90.0, 95.0)),
    }

    state, panel = build_market_stress_panel(frames)

    assert state == "Stressed"
    assert state in MARKET_STATES
    assert "below its 200DMA" in panel.loc[0, "Evidence"]
    assert panel.loc[0, "QQQ 60d RS vs SPY"] < 0


def test_theme_stress_uses_soxx_and_smh_relative_to_qqq() -> None:
    frames = {
        "QQQ": frame(ramp(100.0, 125.0)),
        "SOXX": frame([140.0] * 200 + [100.0] * 61),
        "SMH": frame([130.0] * 200 + [95.0] * 61),
    }

    state, panel = build_theme_stress_panel(frames)

    assert state == "Stressed"
    assert state in THEME_STATES
    assert panel["60d RS vs QQQ"].lt(0).all()


def test_single_name_health_classifies_price_only_trend_states() -> None:
    frames = {
        "QQQ": frame(ramp(100.0, 120.0)),
        "SOXX": frame(ramp(100.0, 130.0)),
        "HEALTHY": frame(ramp(100.0, 150.0)),
        "DAMAGED": frame([120.0] * 200 + [90.0] * 61),
        "EXTENDED": frame([100.0] * 241 + [130.0] * 20),
    }

    healthy = classify_single_name_health(frames, "HEALTHY")
    damaged = classify_single_name_health(frames, "DAMAGED")
    extended = classify_single_name_health(frames, "EXTENDED")

    assert healthy["State"] == "Healthy trend"
    assert damaged["State"] == "Risk reduction candidate"
    assert extended["State"] == "Extended, do not chase"
    assert {healthy["State"], damaged["State"], extended["State"]}.issubset(SINGLE_NAME_STATES)


def test_risk_cockpit_memo_keeps_ml_audit_only_and_no_sizing_change() -> None:
    frames = {
        "SPY": frame(ramp(100.0, 115.0)),
        "QQQ": frame(ramp(100.0, 120.0)),
        "SOXX": frame(ramp(100.0, 125.0)),
        "SMH": frame(ramp(100.0, 123.0)),
        "NVDA": frame(ramp(100.0, 145.0)),
    }

    cockpit = build_risk_cockpit(frames, ["NVDA"])

    assert cockpit.market_state == "Normal"
    assert cockpit.theme_state == "Healthy"
    assert "ML remains audit-only" in cockpit.memo
    assert "sizing instruction" in cockpit.memo
    assert cockpit.single_name_health.loc[0, "State"] == "Healthy trend"
