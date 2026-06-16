"""Headless Research Lab diagnostics assembly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.data.fetch import load_daily_data
from src.decision.config import load_decision_config
from src.features.fourier import rolling_fourier_features
from src.features.regime import classify_regime
from src.features.technical import build_technical_features
from src.features.wavelet import rolling_wavelet_features
from src.ml.datasets import build_supervised_panel, feature_group_columns
from src.ml.diagnostics import (
    build_ml_diagnostics,
    build_ml_feature_audit,
    build_ml_feature_signal_diagnostics,
    build_ml_label_audit,
)
from src.ml.models import MODEL_OPTIONS
from src.ml.target_diagnostics import (
    add_target_candidate_labels,
    build_target_arena_comparison,
    build_target_balance_diagnostics,
    build_target_feature_group_comparison,
    build_target_quality_summary,
    build_target_regime_comparison,
    build_target_stability_summary,
    build_target_walk_forward_comparison,
    target_candidate_registry,
    target_definition_table,
)
from src.ml.validation import compare_feature_groups, summarize_model_selection, walk_forward_validate_classifier
from src.research.export import build_research_lab_export_payload
from src.utils.config import FeatureConfig


RESEARCH_FEATURE_GROUPS = ("technical", "technical_fourier", "technical_wavelet", "all")


@dataclass(frozen=True)
class ResearchLabRunConfig:
    """Parameters for one headless Research Lab diagnostics run."""

    benchmark: str = "QQQ"
    feature_group: str = "all"
    model_mode: str = "auto_select"
    model_name: str | None = None
    train_window: int = 504
    test_window: int = 63
    step: int = 63
    embargo: int = 20
    classification_threshold: float = 0.5
    portfolio_name: str = "Headless Research Lab"
    tickers: tuple[str, ...] = ()
    start: str | None = None
    end: str | None = None
    quick: bool = False


def assemble_research_lab_payload(config: ResearchLabRunConfig) -> dict[str, object]:
    """Build the Research Lab export payload without Streamlit session state."""

    decision_config = load_decision_config()
    benchmark = config.benchmark.upper()
    tickers = _resolve_tickers(config, decision_config.default_ticker_universe)
    model_name = config.model_name or decision_config.default_model
    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model: {model_name}")
    if config.feature_group not in RESEARCH_FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {config.feature_group}")

    end_date = _parse_date(config.end) or date.today()
    start_date = _parse_date(config.start) or (end_date - timedelta(days=365 * 3))
    frames = _load_frames(tickers, benchmark, start_date, end_date)
    feature_frames = build_features_for_universe(frames, FeatureConfig(), benchmark)
    benchmark_frame = frames.get(benchmark)
    if benchmark_frame is None or benchmark_frame.empty:
        raise ValueError(f"No benchmark data available for {benchmark}")

    ticker_feature_frames = {ticker: frame for ticker, frame in feature_frames.items() if ticker in tickers}
    supervised = build_supervised_panel(
        ticker_feature_frames,
        benchmark_price=benchmark_frame["Adj Close"],
        horizon=decision_config.default_label_horizon,
    )
    if supervised.empty:
        raise ValueError("No supervised Research Lab panel could be built")

    columns = feature_group_columns(supervised, config.feature_group)
    if not columns:
        raise ValueError(f"No usable feature columns for feature group {config.feature_group}")
    group_options = {name: feature_group_columns(supervised, name) for name in RESEARCH_FEATURE_GROUPS}
    label_horizon = decision_config.default_label_horizon

    outperformance = walk_forward_validate_classifier(
        supervised,
        columns,
        label_column=f"label_outperform_{label_horizon}d",
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    risk = walk_forward_validate_classifier(
        supervised,
        columns,
        label_column=f"label_drawdown_risk_{label_horizon}d",
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    risk_adjusted = walk_forward_validate_classifier(
        supervised,
        columns,
        label_column=f"label_risk_adjusted_outperform_{label_horizon}d",
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    tail_risk = walk_forward_validate_classifier(
        supervised,
        columns,
        label_column=f"label_tail_risk_adjusted_outperform_{label_horizon}d",
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )

    feature_audit = build_ml_feature_audit(
        supervised,
        columns,
        redundancy_candidate_columns=feature_group_columns(
            supervised,
            config.feature_group,
            prune_redundant_complex=False,
        ),
    )
    feature_signal = build_ml_feature_signal_diagnostics(
        supervised,
        columns,
        horizon=label_horizon,
        high_correlation_pairs=feature_audit.high_correlation_pairs,
    )
    label_audit = build_ml_label_audit(supervised, horizon=label_horizon)
    target_candidates = target_candidate_registry(label_horizon)
    target_definitions = target_definition_table(target_candidates)
    target_panel = add_target_candidate_labels(
        supervised,
        benchmark_price=benchmark_frame["Adj Close"],
        base_horizon=label_horizon,
    )
    target_balance = build_target_balance_diagnostics(target_panel, target_candidates)
    target_walk_forward = build_target_walk_forward_comparison(
        target_panel,
        columns,
        target_candidates,
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    target_feature_group = build_target_feature_group_comparison(
        target_panel,
        group_options,
        target_candidates,
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    target_regime = build_target_regime_comparison(
        target_panel,
        columns,
        target_candidates,
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    target_stability = build_target_stability_summary(
        target_feature_group,
        target_regime,
        target_candidates,
    )
    target_quality = build_target_quality_summary(
        target_balance,
        target_walk_forward,
        target_feature_group,
        target_regime,
    )
    target_arena = build_target_arena_comparison(target_quality)
    diagnostics = build_ml_diagnostics(
        outperformance.predictions,
        risk.predictions,
        outperformance.overall_metrics,
        risk.overall_metrics,
        baseline_panel=supervised,
        risk_adjusted_predictions=risk_adjusted.predictions,
        tail_risk_predictions=tail_risk.predictions,
    )
    comparison = compare_feature_groups(
        supervised,
        group_options,
        label_column=f"label_outperform_{label_horizon}d",
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )

    tables = {
        "ml_diagnostics_summary": diagnostics.summary,
        "ml_score_buckets": diagnostics.score_buckets,
        "ml_baseline_comparison": diagnostics.baseline_comparison,
        "ml_probability_direction_check": diagnostics.probability_direction_check,
        "ml_score_direction_diagnostics": diagnostics.score_direction_summary,
        "drawdown_risk_calibration": diagnostics.drawdown_risk_calibration,
        "drawdown_risk_calibration_quality": diagnostics.drawdown_risk_calibration_quality,
        "model_selection_summary": pd.concat(
            [
                summarize_model_selection(outperformance.fold_metrics, "outperformance"),
                summarize_model_selection(risk.fold_metrics, "drawdown_risk"),
            ],
            ignore_index=True,
        ),
        "model_selection_fold_details": fold_details_for_export(
            outperformance=outperformance.fold_metrics,
            drawdown_risk=risk.fold_metrics,
            risk_adjusted_outperform=risk_adjusted.fold_metrics,
            tail_risk_adjusted_outperform=tail_risk.fold_metrics,
        ),
        "feature_group_comparison": comparison,
        "feature_audit_summary": feature_audit.inventory_summary,
        "feature_family_summary": feature_audit.family_summary,
        "feature_redundancy_selection": feature_audit.redundancy_selection_summary,
        "feature_redundancy_dropped_features": feature_audit.redundancy_selection_report,
        "feature_signal_summary": feature_signal.signal_table,
        "label_prevalence_summary": label_audit.prevalence_summary,
        "label_overlap": label_audit.label_overlap,
        "regime_segmented_ml_diagnostics": diagnostics.regime_segmented,
        "target_candidate_definitions": target_definitions,
        "target_balance": target_balance,
        "target_walk_forward_comparison": target_walk_forward,
        "target_feature_group_comparison": target_feature_group,
        "target_regime_comparison": target_regime,
        "target_stability_summary": target_stability,
        "target_quality_summary": target_quality,
        "target_arena_comparison": target_arena,
        "opportunity_risk_joint_validation": diagnostics.opportunity_risk_joint_validation,
    }
    metadata = {
        "app_name": "Stock Signal Lab",
        "runner": "headless_research_lab",
        "benchmark": benchmark,
        "portfolio_name": config.portfolio_name,
        "ticker_count": len(tickers),
        "tickers": list(tickers),
        "feature_group": config.feature_group,
        "model_name": model_name,
        "model_mode": config.model_mode,
        "train_window": config.train_window,
        "test_window": config.test_window,
        "step_size": config.step,
        "embargo_requested": config.embargo,
        "embargo_effective": max(config.embargo, label_horizon),
        "classification_threshold": config.classification_threshold,
        "target_candidates_enabled": True,
        "extended_target_comparison_enabled": True,
        "target_arena_enabled": True,
        "data_start": str(start_date),
        "data_end": str(end_date),
        "quick": config.quick,
    }
    return build_research_lab_export_payload(run_metadata=metadata, tables=tables)


def build_features_for_universe(
    frames: dict[str, pd.DataFrame],
    feature_config: FeatureConfig,
    benchmark: str,
) -> dict[str, pd.DataFrame]:
    """Build the same technical/Fourier/Wavelet feature frame set used by Research Lab."""

    benchmark_frames = {
        name: frames[name]
        for name in dict.fromkeys(("SPY", "QQQ", benchmark.upper()))
        if name in frames
    }
    output: dict[str, pd.DataFrame] = {}
    for ticker, frame in frames.items():
        features = build_technical_features(frame, benchmark_frames=benchmark_frames, config=feature_config)
        features = features.join(
            rolling_fourier_features(
                frame["Adj Close"],
                window=feature_config.fourier_window,
                n_components=feature_config.fourier_components,
                input_mode=feature_config.fourier_input,
            )
        )
        features = features.join(
            rolling_wavelet_features(
                frame["Adj Close"],
                window=feature_config.wavelet_window,
                wavelet=feature_config.wavelet,
                level=feature_config.wavelet_level,
            )
        )
        output[ticker] = classify_regime(features, use_signal_features=True)
    return output


def fold_details_for_export(**fold_tables: pd.DataFrame) -> pd.DataFrame:
    """Combine walk-forward fold details without rerunning validation."""

    frames = []
    for target_name, frame in fold_tables.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            output = frame.copy()
            output.insert(0, "target", target_name)
            frames.append(output)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _resolve_tickers(config: ResearchLabRunConfig, default_tickers: tuple[str, ...]) -> tuple[str, ...]:
    if config.tickers:
        values = list(config.tickers)
    elif config.quick:
        values = ["NVDA", "MSFT", "AAPL", config.benchmark]
    else:
        values = list(default_tickers)
    return tuple(dict.fromkeys(ticker.strip().upper() for ticker in values if ticker.strip()))


def _load_frames(
    tickers: tuple[str, ...],
    benchmark: str,
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    data_tickers = list(tickers)
    for required in (benchmark, "SPY", "QQQ"):
        if required not in data_tickers:
            data_tickers.append(required)
    frames = {}
    errors = {}
    for ticker in data_tickers:
        try:
            frames[ticker] = load_daily_data(ticker, start=str(start), end=str(end), use_cache=True)
        except Exception as exc:
            errors[ticker] = str(exc)
    usable_tickers = [ticker for ticker in tickers if ticker in frames]
    if not usable_tickers:
        raise RuntimeError(f"No requested tickers could be loaded: {errors}")
    if benchmark not in frames:
        raise RuntimeError(f"Benchmark {benchmark} could not be loaded: {errors.get(benchmark, 'missing')}")
    return frames


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)
