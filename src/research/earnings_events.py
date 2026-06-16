"""Research-only loader for optional local earnings event inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EARNINGS_EVENT_COLUMNS = [
    "ticker",
    "earnings_date",
    "report_timing",
    "eps_surprise_pct",
    "revenue_surprise_pct",
]
EARNINGS_EVENT_INPUT_PATHS = (
    Path("data/research_inputs/earnings_events.csv"),
    Path("data/earnings_events.csv"),
)
REPORT_TIMING_VALUES = {"before_open", "after_close", "during_market", ""}


def find_earnings_events_path(root: str | Path = ".") -> Path | None:
    """Return the first supported local earnings-event CSV path if present."""

    base = Path(root)
    for relative_path in EARNINGS_EVENT_INPUT_PATHS:
        candidate = base / relative_path
        if candidate.exists():
            return candidate
    return None


def empty_earnings_events(
    *,
    status: str = "missing",
    reason: str = "No local earnings event CSV was found.",
    source_path: str | None = None,
) -> pd.DataFrame:
    """Return an empty event frame with lightweight status metadata."""

    frame = pd.DataFrame(columns=EARNINGS_EVENT_COLUMNS)
    frame.attrs["earnings_event_status"] = status
    frame.attrs["earnings_event_reason"] = reason
    if source_path is not None:
        frame.attrs["earnings_event_source_path"] = source_path
    return frame


def load_earnings_events(
    path: str | Path | None = None,
    *,
    root: str | Path = ".",
) -> pd.DataFrame:
    """Load optional local earnings events for research diagnostics.

    Only ticker and earnings_date are required. Optional surprise columns are
    parsed when present, but diagnostics never require them.
    """

    source_path = Path(path) if path is not None else find_earnings_events_path(root)
    if source_path is None:
        return empty_earnings_events()
    if not source_path.exists():
        return empty_earnings_events(
            status="missing",
            reason=f"Earnings event CSV was not found at {source_path}.",
            source_path=str(source_path),
        )

    try:
        raw = pd.read_csv(source_path)
    except Exception as exc:
        return empty_earnings_events(
            status="invalid_schema",
            reason=f"Earnings event CSV could not be read: {exc}",
            source_path=str(source_path),
        )

    normalized_columns = {str(column).strip().lower(): column for column in raw.columns}
    required = {"ticker", "earnings_date"}
    missing_required = sorted(required - set(normalized_columns))
    if missing_required:
        return empty_earnings_events(
            status="invalid_schema",
            reason=f"Earnings event CSV is missing required columns: {', '.join(missing_required)}.",
            source_path=str(source_path),
        )

    output = pd.DataFrame()
    for column in EARNINGS_EVENT_COLUMNS:
        if column in normalized_columns:
            output[column] = raw[normalized_columns[column]]
        else:
            output[column] = pd.NA

    output["ticker"] = output["ticker"].astype("string").str.strip().str.upper()
    output["earnings_date"] = pd.to_datetime(output["earnings_date"], errors="coerce").dt.normalize()
    output["report_timing"] = output["report_timing"].fillna("").astype("string").str.strip().str.lower()
    output.loc[~output["report_timing"].isin(REPORT_TIMING_VALUES), "report_timing"] = ""
    for column in ("eps_surprise_pct", "revenue_surprise_pct"):
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=["ticker", "earnings_date"])
    output = output[output["ticker"] != ""]
    output = output.drop_duplicates(subset=["ticker", "earnings_date"], keep="last")
    output = output.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    output.attrs["earnings_event_status"] = "loaded"
    output.attrs["earnings_event_reason"] = "Loaded local earnings event CSV."
    output.attrs["earnings_event_source_path"] = str(source_path)
    return output
