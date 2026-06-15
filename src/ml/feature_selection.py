"""Feature selection helpers for supervised ML research datasets."""

from __future__ import annotations

import re

import pandas as pd


FEATURE_REDUNDANCY_CORRELATION_THRESHOLD = 0.98
FEATURE_REDUNDANCY_REPORT_COLUMNS = [
    "dropped_feature",
    "kept_feature",
    "abs_correlation",
    "reason",
]

DERIVED_SIGNAL_FEATURES = {
    "fourier_cycle_clarity",
    "fourier_cycle_strength",
    "fourier_noise_diffusion",
    "wavelet_clean_trend",
    "wavelet_trend_quality",
    "wavelet_noise_pressure",
    "wavelet_medium_long_energy_share",
}
INTERPRETABLE_RAW_FEATURES = {
    "fourier_energy_concentration",
    "fourier_spectral_entropy",
    "wavelet_trend_return",
    "wavelet_short_noise_intensity",
}
LOW_LEVEL_TRANSFORM_PATTERNS = (
    re.compile(r"^fourier_(amp|freq|period)_\d+$"),
    re.compile(r"^wavelet_energy_scale_\d+$"),
)


def feature_redundancy_priority(column_name: str) -> tuple[int, str]:
    """Return a deterministic preference key for redundant transform features.

    Lower tuples are preferred. Derived, interpretable transform signals rank
    ahead of legacy summary features, which rank ahead of lower-level scale or
    amplitude fields.
    """

    if column_name in DERIVED_SIGNAL_FEATURES:
        return (0, column_name)
    if column_name in INTERPRETABLE_RAW_FEATURES:
        return (1, column_name)
    if any(pattern.match(column_name) for pattern in LOW_LEVEL_TRANSFORM_PATTERNS):
        return (2, column_name)
    return (3, column_name)


def _empty_redundancy_report() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_REDUNDANCY_REPORT_COLUMNS)


def _correlation_eligible_columns(features: pd.DataFrame, candidate_columns: list[str]) -> list[str]:
    eligible: list[str] = []
    for column in candidate_columns:
        if column not in features:
            continue
        if not pd.api.types.is_numeric_dtype(features[column]):
            continue
        values = pd.to_numeric(features[column], errors="coerce").replace(
            [float("inf"), -float("inf")],
            float("nan"),
        )
        valid = values.dropna()
        if valid.empty or valid.nunique(dropna=True) <= 1:
            continue
        eligible.append(column)
    return eligible


def _final_kept_feature(feature: str, replacement_map: dict[str, str]) -> str:
    seen: set[str] = set()
    current = feature
    while current in replacement_map and current not in seen:
        seen.add(current)
        current = replacement_map[current]
    return current


def prune_redundant_features(
    features: pd.DataFrame,
    candidate_columns: list[str],
    correlation_threshold: float = FEATURE_REDUNDANCY_CORRELATION_THRESHOLD,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Prune highly correlated candidate features using an explicit preference rule."""

    candidates = list(dict.fromkeys(candidate_columns))
    eligible = _correlation_eligible_columns(features, candidates)
    if len(eligible) < 2:
        return candidates, [], _empty_redundancy_report()

    data = features[eligible].apply(pd.to_numeric, errors="coerce").replace(
        [float("inf"), -float("inf")],
        float("nan"),
    )
    correlations = data.corr(numeric_only=True).abs()
    pairs: list[tuple[float, str, str]] = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value) and float(value) >= correlation_threshold:
                pairs.append((float(value), left, right))

    if not pairs:
        return candidates, [], _empty_redundancy_report()

    pairs = sorted(
        pairs,
        key=lambda item: (
            -item[0],
            feature_redundancy_priority(item[1]),
            feature_redundancy_priority(item[2]),
            item[1],
            item[2],
        ),
    )
    dropped: set[str] = set()
    rows: list[dict[str, object]] = []
    replacement_map: dict[str, str] = {}
    for abs_correlation, left, right in pairs:
        if left in dropped or right in dropped:
            continue
        left_priority = feature_redundancy_priority(left)
        right_priority = feature_redundancy_priority(right)
        if left_priority <= right_priority:
            kept, removed = left, right
        else:
            kept, removed = right, left
        dropped.add(removed)
        replacement_map[removed] = kept
        rows.append(
            {
                "dropped_feature": removed,
                "kept_feature": kept,
                "abs_correlation": abs_correlation,
                "reason": "Kept the higher-priority Fourier/Wavelet signal for a highly correlated pair.",
            }
        )

    kept_columns = [column for column in candidates if column not in dropped]
    dropped_columns = [column for column in candidates if column in dropped]
    for row in rows:
        row["kept_feature"] = _final_kept_feature(str(row["kept_feature"]), replacement_map)
    report = pd.DataFrame(rows, columns=FEATURE_REDUNDANCY_REPORT_COLUMNS)
    return kept_columns, dropped_columns, report
