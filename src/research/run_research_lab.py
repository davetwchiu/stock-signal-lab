"""Command-line headless Research Lab runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.research.export import export_research_lab_payload
from src.research.lab import ResearchLabRunConfig, assemble_research_lab_payload


def build_parser() -> argparse.ArgumentParser:
    """Return the headless Research Lab argument parser."""

    parser = argparse.ArgumentParser(description="Run Research Lab diagnostics without opening Streamlit.")
    parser.add_argument("--benchmark", default="QQQ")
    parser.add_argument("--feature-group", default="all", choices=["technical", "technical_fourier", "technical_wavelet", "all"])
    parser.add_argument("--model-mode", default="auto_select", choices=["current_default", "auto_select"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--train-window", type=int, default=504)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=20)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--portfolio-name", default="Headless Research Lab")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker list.")
    parser.add_argument("--tickers-file", type=Path, default=None)
    parser.add_argument("--start", default=None, help="Optional YYYY-MM-DD data start.")
    parser.add_argument("--end", default=None, help="Optional YYYY-MM-DD data end.")
    parser.add_argument("--output-root", type=Path, default=Path("data/research_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--export", action="store_true", help="Write the diagnostics bundle.")
    parser.add_argument("--quick", action="store_true", help="Use a reduced default ticker set if --tickers is omitted.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_from_args(args: argparse.Namespace):
    """Run diagnostics from parsed CLI args and optionally export the bundle."""

    tickers = _parse_tickers(args.tickers, args.tickers_file)
    config = ResearchLabRunConfig(
        benchmark=args.benchmark,
        feature_group=args.feature_group,
        model_mode=args.model_mode,
        model_name=args.model_name,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        embargo=args.embargo,
        classification_threshold=args.classification_threshold,
        portfolio_name=args.portfolio_name,
        tickers=tuple(tickers),
        start=args.start,
        end=args.end,
        quick=args.quick,
    )
    payload = assemble_research_lab_payload(config)
    if not args.export:
        print("Research Lab diagnostics assembled. Pass --export to write data/research_runs output.")
        return payload
    result = export_research_lab_payload(
        payload,
        output_root=args.output_root,
        run_id=args.run_name,
    )
    print(f"Research Lab diagnostics bundle: {result.run_dir}")
    print(f"Latest Research Lab diagnostics bundle: {result.latest_dir}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_from_args(args)
    return 0


def _parse_tickers(raw: str | None, tickers_file: Path | None) -> list[str]:
    values: list[str] = []
    if raw:
        values.extend(part.strip().upper() for part in raw.split(",") if part.strip())
    if tickers_file is not None:
        file_text = tickers_file.read_text(encoding="utf-8")
        for line in file_text.splitlines():
            values.extend(part.strip().upper() for part in line.split(",") if part.strip())
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
