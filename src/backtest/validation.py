"""Walk-forward validation scaffolding for future model experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """One train/test split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: pd.Index
    test_index: pd.Index


def walk_forward_splits(
    index: pd.Index,
    train_window: int = 252,
    test_window: int = 63,
    step: int | None = None,
) -> Iterator[WalkForwardSplit]:
    """Yield rolling train/test windows over a date-like index."""

    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive")

    ordered = pd.Index(index).sort_values()
    active_step = step or test_window
    start = 0
    while start + train_window + test_window <= len(ordered):
        train_idx = ordered[start : start + train_window]
        test_idx = ordered[start + train_window : start + train_window + test_window]
        yield WalkForwardSplit(
            train_start=pd.Timestamp(train_idx[0]),
            train_end=pd.Timestamp(train_idx[-1]),
            test_start=pd.Timestamp(test_idx[0]),
            test_end=pd.Timestamp(test_idx[-1]),
            train_index=train_idx,
            test_index=test_idx,
        )
        start += active_step

