"""Display-only shortlist filters for the Decision Cockpit."""

from __future__ import annotations

import re

import pandas as pd


SHORTLIST_VIEW_ALL = "All"
SHORTLIST_VIEW_STRONG = "Strong setups"
SHORTLIST_VIEW_WATCHLIST = "Watchlist / wait for pullback"
SHORTLIST_VIEW_PULLBACK = "High pullback risk"
SHORTLIST_VIEW_WEAK = "Weak or avoid"
SHORTLIST_VIEW_OPTIONS = [
    SHORTLIST_VIEW_ALL,
    SHORTLIST_VIEW_STRONG,
    SHORTLIST_VIEW_WATCHLIST,
    SHORTLIST_VIEW_PULLBACK,
    SHORTLIST_VIEW_WEAK,
]

STRONG_SCORE_THRESHOLD = 70
LOW_SCORE_THRESHOLD = 40
HIGH_PULLBACK_RISK_THRESHOLD = 0.60
ELEVATED_PULLBACK_RISK_THRESHOLD = 0.40
STRONG_RELATIVE_STRENGTH_RANK = 3

_ACTION_AVOID_WORDS = ("avoid", "reduce", "sell", "trim", "exit")
_ACTION_WAIT_WORDS = ("watch", "wait", "hold")


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).lower())


def _find_column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized_columns = {_normalize_column_name(column): column for column in table.columns}
    for candidate in candidates:
        column = normalized_columns.get(_normalize_column_name(candidate))
        if column is not None:
            return str(column)
    return None


def _empty_mask(table: pd.DataFrame, value: bool = False) -> pd.Series:
    return pd.Series(value, index=table.index, dtype=bool)


def _numeric_series(table: pd.DataFrame, column: str | None, scale_percent: bool = False) -> pd.Series:
    if column is None:
        return pd.Series(pd.NA, index=table.index, dtype="Float64")
    raw = table[column]
    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
    else:
        numeric = pd.to_numeric(raw.astype(str).str.replace("%", "", regex=False), errors="coerce")
    if scale_percent:
        numeric = numeric.where(numeric <= 1, numeric / 100)
    return numeric


def _text_mask(table: pd.DataFrame, column: str | None, words: tuple[str, ...]) -> pd.Series:
    if column is None:
        return _empty_mask(table)
    text = table[column].fillna("").astype(str).str.lower()
    return text.str.contains("|".join(re.escape(word) for word in words), regex=True)


def _pullback_risk_mask(table: pd.DataFrame, threshold: float) -> pd.Series:
    risk_column = _find_column(
        table,
        (
            "Drawdown-risk probability",
            "ML Drawdown-Risk Probability",
            "Pullback risk",
            "Drawdown risk probability",
        ),
    )
    numeric_risk = _numeric_series(table, risk_column, scale_percent=True)
    high_text_risk = _text_mask(table, risk_column, ("high",))
    return numeric_risk.ge(threshold).fillna(False) | high_text_risk


def filter_decision_shortlist(table: pd.DataFrame, view: str) -> pd.DataFrame:
    """Return rows for a display-only Decision Cockpit shortlist view."""

    if table.empty or view == SHORTLIST_VIEW_ALL:
        return table.copy()

    score_column = _find_column(table, ("ML score", "ML Score", "Opportunity score"))
    action_column = _find_column(table, ("Suggested action", "Suggested Action", "Action"))
    rs_column = _find_column(table, ("Relative strength rank", "Relative Strength Rank"))

    score = _numeric_series(table, score_column)
    rs_rank = _numeric_series(table, rs_column)
    high_score = score.ge(STRONG_SCORE_THRESHOLD).fillna(False)
    low_score = score.le(LOW_SCORE_THRESHOLD).fillna(False)
    strong_rs = rs_rank.le(STRONG_RELATIVE_STRENGTH_RANK).fillna(False)
    avoid_action = _text_mask(table, action_column, _ACTION_AVOID_WORDS)
    wait_action = _text_mask(table, action_column, _ACTION_WAIT_WORDS)
    high_risk = _pullback_risk_mask(table, HIGH_PULLBACK_RISK_THRESHOLD)
    elevated_risk = _pullback_risk_mask(table, ELEVATED_PULLBACK_RISK_THRESHOLD)

    if view == SHORTLIST_VIEW_STRONG:
        mask = high_score & ~high_risk & ~avoid_action & ~wait_action
    elif view == SHORTLIST_VIEW_WATCHLIST:
        mask = (high_score | strong_rs) & (elevated_risk | wait_action)
    elif view == SHORTLIST_VIEW_PULLBACK:
        mask = high_risk
    elif view == SHORTLIST_VIEW_WEAK:
        mask = avoid_action | low_score
    else:
        mask = _empty_mask(table)

    return table.loc[mask].copy()
