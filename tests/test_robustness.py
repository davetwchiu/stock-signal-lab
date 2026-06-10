from __future__ import annotations

import pandas as pd

from src.features.regime import UPTREND_LOW_VOL
from src.robustness.runner import run_robustness_tests
from src.robustness.ablation import conclusion_from_metrics
from src.robustness.stability import stability_score, summarize_robustness, warning_flags


def test_stability_summary_and_warning_flags() -> None:
    results = pd.DataFrame({"Sharpe": [0.8, 0.7, -0.2, 1.8]})

    summary = summarize_robustness(results, metric="Sharpe")

    assert 0 <= stability_score(results["Sharpe"]) <= 100
    assert not summary.empty
    assert warning_flags(results, "Sharpe")


def test_ablation_conclusion_fields() -> None:
    metrics = pd.Series({"roc_auc": 0.60, "f1": 0.50, "pr_auc": 0.55})
    quintiles = pd.DataFrame({"score_quintile": ["Q1", "Q5"], "average_forward_excess_return": [-0.01, 0.02]})

    assert conclusion_from_metrics(metrics, quintiles) == "adds value"


def test_robustness_runner_smoke() -> None:
    index = pd.date_range("2020-01-01", periods=90, freq="B")
    price = pd.Series(range(100, 190), index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "Adj Close": price,
            "return_20d": price.pct_change(20),
            "volatility_20d": price.pct_change().rolling(20).std(),
            "regime": UPTREND_LOW_VOL,
        },
        index=index,
    )
    price_frame = pd.DataFrame({"Adj Close": price}, index=index)

    results, summary = run_robustness_tests(
        feature_frames={"AAA": frame, "SPY": frame},
        price_frames={"AAA": price_frame, "SPY": price_frame},
        benchmark_frames={"SPY": price_frame},
        feature_groups=["technical"],
        horizons=[10],
        ml_thresholds=[0.5],
        drawdown_risk_thresholds=[0.5],
        transaction_costs_bps=[0.0],
        slippage_bps_values=[0.0],
        train_windows=[30],
        test_window=10,
        step=10,
        embargo=5,
    )

    assert not results.empty
    assert not summary.empty
