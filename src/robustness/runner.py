"""Robustness test runner for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from src.ml.datasets import build_supervised_panel, feature_group_columns
from src.ml.scoring import ml_score
from src.ml.validation import deduplicate_prediction_keys, prediction_merge_keys, walk_forward_validate_classifier
from src.portfolio.allocation import AllocationConfig
from src.portfolio.risk import RiskControlConfig
from src.portfolio.simulator import simulate_portfolio
from src.robustness.stability import summarize_robustness, warning_flags


@dataclass(frozen=True)
class RobustnessCase:
    """One robustness assumption set."""

    feature_group: str
    horizon: int
    benchmark: str
    ml_threshold: float
    drawdown_threshold: float
    transaction_cost_bps: float
    slippage_bps: float
    train_window: int


def _score_panel_from_predictions(out_predictions: pd.DataFrame, risk_predictions: pd.DataFrame) -> pd.DataFrame:
    keys = prediction_merge_keys(out_predictions, risk_predictions)
    out = deduplicate_prediction_keys(out_predictions, keys)
    risk = deduplicate_prediction_keys(risk_predictions, keys)
    merged = out.merge(
        risk,
        on=keys,
        suffixes=("_out", "_risk"),
    )
    if merged.empty:
        return pd.DataFrame()
    out_prob = merged["probability_out"]
    risk_prob = merged["probability_risk"]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(merged["Date"]),
            "Ticker": merged["Ticker"],
            **({"fold": merged["fold"]} if "fold" in merged else {}),
            "ML Outperformance Probability": out_prob,
            "ML Drawdown-Risk Probability": risk_prob,
            "ML Score": ml_score(out_prob, risk_prob),
        }
    )


def run_robustness_tests(
    feature_frames: dict[str, pd.DataFrame],
    price_frames: dict[str, pd.DataFrame],
    benchmark_frames: dict[str, pd.DataFrame],
    feature_groups: list[str],
    horizons: list[int],
    ml_thresholds: list[float],
    drawdown_risk_thresholds: list[float],
    transaction_costs_bps: list[float],
    slippage_bps_values: list[float],
    model_name: str = "logistic_regression",
    train_windows: list[int] | None = None,
    test_window: int = 63,
    step: int | None = None,
    embargo: int = 20,
    benchmark_choices: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a compact grid of validation and portfolio robustness tests."""

    rows: list[dict[str, object]] = []
    active_train_windows = train_windows or [252]
    active_benchmarks = benchmark_choices or list(benchmark_frames)

    for (
        benchmark,
        horizon,
        feature_group,
        ml_threshold,
        risk_threshold,
        cost_bps,
        slippage_bps,
        train_window,
    ) in product(
        active_benchmarks,
        horizons,
        feature_groups,
        ml_thresholds,
        drawdown_risk_thresholds,
        transaction_costs_bps,
        slippage_bps_values,
        active_train_windows,
    ):
        benchmark_frame = benchmark_frames.get(benchmark)
        if benchmark_frame is None or benchmark_frame.empty:
            continue
        dataset = build_supervised_panel(
            feature_frames,
            benchmark_price=benchmark_frame["Adj Close"],
            horizon=horizon,
            drawdown_threshold=-abs(risk_threshold),
        )
        columns = feature_group_columns(dataset, feature_group)
        if not columns:
            continue
        out_label = f"label_outperform_{horizon}d"
        risk_label = f"label_drawdown_risk_{horizon}d"
        out_result = walk_forward_validate_classifier(
            dataset,
            columns,
            out_label,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=ml_threshold,
        )
        risk_result = walk_forward_validate_classifier(
            dataset,
            columns,
            risk_label,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=risk_threshold,
        )
        score_panel = _score_panel_from_predictions(out_result.predictions, risk_result.predictions)
        portfolio = simulate_portfolio(
            price_frames,
            score_panel,
            allocation_config=AllocationConfig(drawdown_risk_threshold=risk_threshold),
            risk_config=RiskControlConfig(),
            benchmark_features=feature_frames.get(benchmark),
            benchmark_price=benchmark_frame["Adj Close"],
            transaction_cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        )
        if out_result.overall_metrics.empty or portfolio.summary.empty:
            continue
        row = out_result.overall_metrics.iloc[0].to_dict()
        row.update(portfolio.summary.iloc[0].add_prefix("portfolio_").to_dict())
        row.update(
            {
                "benchmark": benchmark,
                "horizon": horizon,
                "feature_group": feature_group,
                "ml_threshold": ml_threshold,
                "drawdown_risk_threshold": risk_threshold,
                "transaction_cost_bps": cost_bps,
                "slippage_bps": slippage_bps,
                "train_window": train_window,
            }
        )
        rows.append(row)

    results = pd.DataFrame(rows)
    metric = "portfolio_Sharpe"
    if not results.empty and (metric not in results or results[metric].dropna().empty):
        metric = "portfolio_CAGR"
    summary = summarize_robustness(results, metric=metric) if not results.empty else pd.DataFrame()
    if not summary.empty:
        summary["warning_flags"] = ", ".join(warning_flags(results, metric))
    return results, summary
