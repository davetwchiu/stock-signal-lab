"""Command-line runner for the research-only simple ROI backtest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.research.backtest import SimpleROIBacktestConfig, default_lookback_start, run_simple_roi_backtest, write_optional_csv


def build_parser() -> argparse.ArgumentParser:
    """Return the simple ROI backtest argument parser."""

    parser = argparse.ArgumentParser(description="Run research-only simple strategy ROI backtests.")
    parser.add_argument("--tickers", default="XLK,QQQ,NVDA,TSM,PLTR", help="Comma-separated ticker list.")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-06-17")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--lookback-start", default=None)
    parser.add_argument("--starting-capital", type=float, default=1_000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--train-window", type=int, default=504)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=20)
    parser.add_argument("--profile", default="Balanced")
    parser.add_argument("--feature-group", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    """Run the backtest from parsed CLI args."""

    tickers = tuple(part.strip().upper() for part in args.tickers.split(",") if part.strip())
    config = SimpleROIBacktestConfig(
        tickers=tickers,
        start=args.start,
        end=args.end,
        benchmark=args.benchmark.upper(),
        lookback_start=args.lookback_start or default_lookback_start(args.start),
        starting_capital=args.starting_capital,
        transaction_cost_bps=args.transaction_cost_bps,
        model_name=args.model_name,
        feature_group=args.feature_group,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        embargo=args.embargo,
        profile_name=args.profile,
        use_cache=not args.no_cache,
    )
    results = run_simple_roi_backtest(config)
    write_optional_csv(results, args.output)
    print_results(results, args.output)
    return results


def print_results(results: pd.DataFrame, output: Path | None = None) -> None:
    """Print a compact evidence table for terminal use."""

    if results.empty:
        print("No backtest rows produced.")
        return
    display = results[
        [
            "ticker",
            "strategy",
            "final_value",
            "total_return",
            "annualized_return",
            "max_drawdown",
            "number_of_trades",
            "days_in_market",
            "percent_days_in_market",
            "worst_drawdown_start",
            "worst_drawdown_end",
            "beats_buy_hold",
            "beats_200dma",
            "improves_return_drawdown_tradeoff_vs_ssl",
            "best_strategy_per_ticker",
            "ssl_beats_buy_hold",
            "ssl_beats_200dma",
            "ml_adds_value_vs_non_ml_app_rule",
        ]
    ].copy()
    for column in ("final_value", "total_return", "annualized_return", "max_drawdown", "percent_days_in_market"):
        display[column] = pd.to_numeric(display[column], errors="coerce").round(4)
    print(display.to_string(index=False))
    winners = results.drop_duplicates("ticker")[["ticker", "best_strategy_per_ticker", "worst_strategy_per_ticker"]]
    print("\nPer-ticker winners:")
    print(winners.to_string(index=False))
    ssl = results[results["strategy"] == "Stock Signal Lab production rule"]
    print("\nStock Signal Lab checks:")
    print(
        ssl[
            [
                "ticker",
                "ssl_beats_buy_hold",
                "ssl_beats_200dma",
                "ml_adds_value_vs_non_ml_app_rule",
                "max_drawdown",
                "final_value",
            ]
        ].to_string(index=False)
    )
    if output is not None:
        print(f"\nCSV output: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
