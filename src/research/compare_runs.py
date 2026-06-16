"""CLI for comparing exported Research Lab bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.research.iteration import compare_research_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two exported Research Lab bundles.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--objective", default="ml_target_research")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_research_runs(args.baseline, args.candidate, objective=args.objective)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
