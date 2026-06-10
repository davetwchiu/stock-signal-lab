from __future__ import annotations

import pandas as pd

from src.portfolio.reporting import generate_decision_report


def test_decision_report_generation() -> None:
    current_scores = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Rule-Based Regime": ["Uptrend / low volatility"],
            "ML Score": [90.0],
            "ML Drawdown-Risk Probability": [0.10],
            "Confidence": ["High"],
        }
    )
    latest_features = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Adj Close": [100.0],
            "rs_spy_60d": [0.05],
            "dist_ma_50d": [0.02],
            "dist_ma_200d": [0.03],
            "volume_z_20d": [0.0],
        }
    )

    report = generate_decision_report(current_scores, latest_features, benchmark_regime="Uptrend / low volatility")

    assert report.ticker_table.iloc[0]["Suggested Action"] == "Add"
    assert "Ticker Table" in report.markdown
    assert "<br>" in report.html

