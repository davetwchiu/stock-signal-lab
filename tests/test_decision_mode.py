from __future__ import annotations

import pandas as pd

from src.decision.config import DEFAULT_ADVANCED_OVERRIDE, load_decision_config, profile_settings
from src.decision.report import generate_markdown_report, portfolio_summary_text
from src.decision.shortlist import (
    SHORTLIST_VIEW_ALL,
    SHORTLIST_VIEW_PULLBACK,
    SHORTLIST_VIEW_STRONG,
    SHORTLIST_VIEW_WATCHLIST,
    SHORTLIST_VIEW_WEAK,
    filter_decision_shortlist,
)
from src.decision.table import (
    build_decision_table,
    confidence_from_score,
    one_line_reason,
    parse_current_weights_input,
    target_exposure_bucket,
)
from src.features.regime import DOWNTREND_HIGH_RISK, UPTREND_LOW_VOL
from src.portfolio.allocation import suggested_action


def test_default_decision_config_loading() -> None:
    config = load_decision_config()

    assert "NVDA" in config.default_ticker_universe
    assert config.default_benchmark == "SPY"
    assert config.default_model == "logistic_regression"
    assert profile_settings(config, "Balanced").cash_floor == config.default_cash_floor


def test_action_label_generation() -> None:
    assert suggested_action(0.00, 0.05) == "Add"
    assert suggested_action(0.05, 0.05) == "Hold"
    assert suggested_action(0.10, 0.05) == "Trim"
    assert suggested_action(0.10, 0.00) == "Exit"
    assert suggested_action(0.00, 0.00) == "Watch"


def test_target_exposure_bucket_generation() -> None:
    assert target_exposure_bucket(0.00, 0.12) == "0%"
    assert target_exposure_bucket(0.03, 0.12) == "25%"
    assert target_exposure_bucket(0.06, 0.12) == "50%"
    assert target_exposure_bucket(0.09, 0.12) == "75%"
    assert target_exposure_bucket(0.12, 0.12) == "100%"


def test_confidence_bucket_generation() -> None:
    assert confidence_from_score(50, 0.50) == "Low"
    assert confidence_from_score(65, 0.50) == "Medium"
    assert confidence_from_score(90, 0.10) == "High"


def test_one_line_reason_includes_decision_drivers() -> None:
    row = pd.Series(
        {
            "Suggested action": "Add",
            "Target exposure bucket": "100%",
            "Rule-based regime": UPTREND_LOW_VOL,
            "ML score": 90,
            "Drawdown-risk probability": 0.10,
            "Relative strength rank": 1.0,
        }
    )

    reason = one_line_reason(row)

    assert reason == (
        "Add toward 100% of max position; bullish low-volatility trend; "
        "very high ML score 90; low drawdown risk 10%; relative strength rank #1."
    )


def test_one_line_reason_explains_weak_watch_setup() -> None:
    row = pd.Series(
        {
            "Suggested action": "Watch",
            "Target exposure bucket": "0%",
            "Rule-based regime": DOWNTREND_HIGH_RISK,
            "ML score": 30,
            "Drawdown-risk probability": 0.72,
        }
    )

    reason = one_line_reason(row)

    assert reason == (
        "Watch; target 0% exposure; weak downtrend regime; "
        "weak ML score 30; high drawdown risk 72%."
    )


def test_decision_mode_table_output_shape() -> None:
    config = load_decision_config()
    profile = profile_settings(config, "Balanced")
    current_scores = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Rule-Based Regime": [UPTREND_LOW_VOL, UPTREND_LOW_VOL],
            "ML Score": [90.0, 30.0],
            "ML Drawdown-Risk Probability": [0.10, 0.70],
        }
    )
    latest_features = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Adj Close": [100.0, 50.0],
            "rs_spy_60d": [0.05, -0.02],
        }
    )

    table = build_decision_table(current_scores, latest_features, config, profile)

    assert list(table.columns) == [
        "Ticker",
        "Price",
        "Rule-based regime",
        "ML score",
        "Drawdown-risk probability",
        "Relative strength rank",
        "Suggested action",
        "Target exposure bucket",
        "Confidence",
        "One-line reason",
    ]
    assert len(table) == 2


def test_decision_table_uses_active_benchmark_relative_strength_rank() -> None:
    config = load_decision_config()
    profile = profile_settings(config, "Balanced")
    current_scores = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Rule-Based Regime": [UPTREND_LOW_VOL, UPTREND_LOW_VOL],
            "ML Score": [90.0, 89.0],
            "ML Drawdown-Risk Probability": [0.10, 0.10],
        }
    )
    latest_features = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Adj Close": [100.0, 50.0],
            "rs_spy_60d": [0.10, -0.10],
            "rs_smh_60d": [-0.05, 0.20],
        }
    )

    table = build_decision_table(current_scores, latest_features, config, profile, benchmark="SMH")
    ranks = table.set_index("Ticker")["Relative strength rank"]

    assert ranks.loc["BBB"] == 1
    assert ranks.loc["AAA"] == 2


def test_decision_table_falls_back_to_spy_relative_strength_rank() -> None:
    config = load_decision_config()
    profile = profile_settings(config, "Balanced")
    current_scores = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Rule-Based Regime": [UPTREND_LOW_VOL, UPTREND_LOW_VOL],
            "ML Score": [90.0, 89.0],
            "ML Drawdown-Risk Probability": [0.10, 0.10],
        }
    )
    latest_features = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Adj Close": [100.0, 50.0],
            "rs_spy_60d": [0.10, -0.10],
        }
    )

    table = build_decision_table(current_scores, latest_features, config, profile, benchmark="SOXX")
    ranks = table.set_index("Ticker")["Relative strength rank"]

    assert ranks.loc["AAA"] == 1
    assert ranks.loc["BBB"] == 2


def test_decision_table_sorts_actions_in_intended_order() -> None:
    config = load_decision_config()
    profile = profile_settings(config, "Balanced")
    current_scores = pd.DataFrame(
        {
            "Ticker": ["WATCH", "EXIT", "TRIM", "HOLD", "ADD"],
            "Rule-Based Regime": [UPTREND_LOW_VOL] * 5,
            "ML Score": [20.0, 95.0, 94.0, 93.0, 92.0],
            "ML Drawdown-Risk Probability": [0.70, 0.70, 0.10, 0.10, 0.10],
        }
    )
    latest_features = pd.DataFrame(
        {
            "Ticker": ["WATCH", "EXIT", "TRIM", "HOLD", "ADD"],
            "Adj Close": [100.0] * 5,
            "rs_spy_60d": [0.01] * 5,
        }
    )
    current_weights = pd.Series({"EXIT": 0.10, "TRIM": 0.14, "HOLD": 0.12, "ADD": 0.00, "WATCH": 0.00})

    table = build_decision_table(current_scores, latest_features, config, profile, current_weights=current_weights)

    assert table["Suggested action"].tolist() == ["Add", "Hold", "Trim", "Exit", "Watch"]


def test_parse_current_weights_input_accepts_newlines_commas_and_equals() -> None:
    weights = parse_current_weights_input("nvda 0.12\nTSLA=0.08, pltr 0.05")

    assert weights.to_dict() == {"NVDA": 0.12, "TSLA": 0.08, "PLTR": 0.05}


def test_parse_current_weights_input_rejects_malformed_entries() -> None:
    try:
        parse_current_weights_input("NVDA")
    except ValueError as error:
        assert "TICKER weight" in str(error)
    else:
        raise AssertionError("Expected malformed current weights to raise ValueError")


def test_decision_shortlist_filters_existing_table_columns() -> None:
    table = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "ML score": [82, 78, 65, 32],
            "Drawdown-risk probability": [0.20, 0.55, 0.72, 0.30],
            "Relative strength rank": [1, 2, 4, 5],
            "Suggested action": ["Add", "Watch", "Hold", "Trim"],
        }
    )

    assert filter_decision_shortlist(table, SHORTLIST_VIEW_ALL)["Ticker"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
    assert filter_decision_shortlist(table, SHORTLIST_VIEW_STRONG)["Ticker"].tolist() == ["AAA"]
    assert filter_decision_shortlist(table, SHORTLIST_VIEW_WATCHLIST)["Ticker"].tolist() == ["BBB"]
    assert filter_decision_shortlist(table, SHORTLIST_VIEW_PULLBACK)["Ticker"].tolist() == ["CCC"]
    assert filter_decision_shortlist(table, SHORTLIST_VIEW_WEAK)["Ticker"].tolist() == ["DDD"]


def test_decision_shortlist_filters_display_labels_and_percent_strings() -> None:
    table = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "Opportunity score": ["75", "39", "88"],
            "Pullback risk": ["25%", "68%", "High"],
            "Relative strength rank": ["2", "5", "1"],
            "Suggested action": ["Add", "Avoid", "Add"],
        }
    )

    assert filter_decision_shortlist(table, SHORTLIST_VIEW_STRONG)["Ticker"].tolist() == ["AAA"]
    assert filter_decision_shortlist(table, SHORTLIST_VIEW_WATCHLIST)["Ticker"].tolist() == ["CCC"]
    assert filter_decision_shortlist(table, SHORTLIST_VIEW_PULLBACK)["Ticker"].tolist() == ["BBB", "CCC"]
    assert filter_decision_shortlist(table, SHORTLIST_VIEW_WEAK)["Ticker"].tolist() == ["BBB"]


def test_markdown_report_generation() -> None:
    table = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Suggested action": ["Add"],
            "Target exposure bucket": ["100%"],
            "Confidence": ["High"],
            "One-line reason": ["Strong score with acceptable risk."],
            "Drawdown-risk probability": [0.10],
            "Rule-based regime": [UPTREND_LOW_VOL],
        }
    )
    summary = portfolio_summary_text(UPTREND_LOW_VOL, table, 0.10)
    report = generate_markdown_report(
        table,
        profile="Balanced",
        benchmark="SPY",
        market_regime=UPTREND_LOW_VOL,
        suggested_gross_exposure=0.90,
        suggested_cash_level=0.10,
        summary_text=summary,
    )

    assert "# Stock Signal Lab Decision Report" in report
    assert "## Actions" in report
    assert "AAA" in report


def test_advanced_override_hidden_by_default() -> None:
    assert DEFAULT_ADVANCED_OVERRIDE is False
