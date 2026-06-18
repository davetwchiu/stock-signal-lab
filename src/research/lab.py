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
    DEFAULT_REGIME_COLUMNS,
    build_drawdown_risk_prevalence_baseline_comparison,
    build_feature_family_importance_stability,
    build_feature_importance_production_readiness,
    build_feature_importance_stability,
    build_ml_diagnostics,
    build_ml_feature_audit,
    build_ml_feature_signal_diagnostics,
    build_ml_label_audit,
    build_validation_fold_stability,
    build_validation_leakage_diagnostics,
    build_validation_overfit_warnings,
)
from src.ml.metrics import classification_metrics
from src.ml.models import MODEL_OPTIONS
from src.ml.target_diagnostics import (
    add_target_candidate_labels,
    build_target_arena_comparison,
    build_target_balance_diagnostics,
    build_target_feature_group_comparison,
    build_target_quality_summary,
    build_target_regime_comparison,
    build_target_stability_summary,
    build_target_stop_rule_comparison,
    build_target_walk_forward_comparison,
    target_candidate_registry,
    target_definition_table,
)
from src.ml.validation import compare_feature_groups, summarize_model_selection, walk_forward_validate_classifier
from src.research.earnings_events import load_earnings_events
from src.research.export import build_research_lab_export_payload
from src.research.portfolio_crowding import build_portfolio_crowding_diagnostics
from src.utils.config import FeatureConfig


RESEARCH_FEATURE_GROUPS = ("technical", "technical_fourier", "technical_wavelet", "all")
RISK_INCREMENTAL_VALUE_COLUMNS = [
    "feature_group",
    "features",
    "sample_count",
    "ticker_count",
    "fold_count",
    "event_prevalence",
    "model_roc_auc",
    "model_pr_auc",
    "model_brier_score",
    "model_calibration_gap",
    "global_fold_baseline_roc_auc",
    "global_fold_baseline_pr_auc",
    "global_fold_baseline_brier_score",
    "global_fold_baseline_calibration_gap",
    "regime_fold_baseline_roc_auc",
    "regime_fold_baseline_pr_auc",
    "regime_fold_baseline_brier_score",
    "regime_fold_baseline_calibration_gap",
    "model_vs_global_fold_baseline",
    "model_vs_regime_fold_baseline",
    "worst_regime",
    "worst_fold",
    "worst_ticker",
    "fallback_count",
    "fold_train_prevalence_details",
    "classification",
]
OPPORTUNITY_BASELINE_CHALLENGE_COLUMNS = [
    "comparator",
    "sample_count",
    "ticker_count",
    "fold_count",
    "event_prevalence",
    "mean_predicted_opportunity",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "calibration_gap",
    "worst_regime",
    "worst_fold",
    "worst_ticker",
    "fallback_count",
    "momentum_feature",
    "bucket_count",
    "fallback_rule",
    "fold_train_prevalence_details",
    "classification",
]
OPPORTUNITY_LABEL_BASELINE_CHALLENGE_COLUMNS = [
    "target_id",
    "display_name",
    "label_column",
    *OPPORTUNITY_BASELINE_CHALLENGE_COLUMNS,
]
OPPORTUNITY_LABEL_BASELINE_BREAKDOWN_COLUMNS = [
    "target_id",
    "display_name",
    "label_column",
    "breakdown",
    "segment",
    *OPPORTUNITY_BASELINE_CHALLENGE_COLUMNS,
]
RISK_ADJUSTED_OPPORTUNITY_FRAGILITY_COLUMNS = [
    "target_id",
    "display_name",
    "label_column",
    "view",
    "segment",
    "excluded_tickers",
    "sample_count",
    "ticker_count",
    "fold_count",
    "event_prevalence",
    "model_roc_auc",
    "model_pr_auc",
    "model_brier_score",
    "model_calibration_gap",
    "model_bucket_spread",
    "inverted_roc_auc",
    "inverted_pr_auc",
    "inverted_brier_score",
    "inverted_calibration_gap",
    "inverted_bucket_spread",
    "outcome_column",
    "normal_top_bucket_forward_outcome",
    "normal_bottom_bucket_forward_outcome",
    "normal_forward_outcome_spread",
    "inverted_top_bucket_forward_outcome",
    "inverted_bottom_bucket_forward_outcome",
    "inverted_forward_outcome_spread",
    "baseline_loss_count",
    "baseline_match_count",
    "model_win_count",
    "unstable_count",
    "worst_ticker",
    "worst_fold",
    "worst_regime",
    "ticker_mix",
    "classification",
]
MOMENTUM_BASELINE_CANDIDATES = (
    "momentum_60d",
    "rs_qqq_60d",
    "rs_spy_60d",
    "momentum_20d",
    "rs_qqq_120d",
    "rs_spy_120d",
)
ADVERSE_OUTCOME_LABEL_COMPARISON_COLUMNS = [
    "label",
    "definition",
    "threshold",
    "sample_count",
    "ticker_count",
    "fold_count",
    "event_prevalence",
    "current_label_overlap_rate",
    "regime_concentration",
    "most_concentrated_regime",
    "model_roc_auc",
    "model_pr_auc",
    "model_brier_score",
    "model_calibration_gap",
    "global_fold_baseline_brier_score",
    "global_fold_baseline_calibration_gap",
    "regime_fold_baseline_brier_score",
    "regime_fold_baseline_calibration_gap",
    "model_vs_global_fold_baseline",
    "model_vs_regime_fold_baseline",
    "worst_regime",
    "worst_fold",
    "worst_ticker",
    "classification",
]


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
    target_stop_rule = build_target_stop_rule_comparison(
        target_quality,
        target_regime,
        target_feature_group,
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
        earnings_events=load_earnings_events(),
    )
    fold_details = fold_details_for_export(
        outperformance=outperformance.fold_metrics,
        drawdown_risk=risk.fold_metrics,
        risk_adjusted_outperform=risk_adjusted.fold_metrics,
        tail_risk_adjusted_outperform=tail_risk.fold_metrics,
    )
    fold_importance = fold_details_for_export(
        outperformance=outperformance.fold_feature_importance,
        drawdown_risk=risk.fold_feature_importance,
        risk_adjusted_outperform=risk_adjusted.fold_feature_importance,
        tail_risk_adjusted_outperform=tail_risk.fold_feature_importance,
    )
    feature_importance_stability = build_feature_importance_stability(fold_importance)
    feature_family_importance_stability = build_feature_family_importance_stability(fold_importance)
    feature_importance_production_readiness = build_feature_importance_production_readiness(
        feature_importance_stability
    )
    validation_leakage = build_validation_leakage_diagnostics(
        fold_details,
        label_horizon_days=label_horizon,
    )
    validation_fold_stability = build_validation_fold_stability(fold_details)
    drawdown_risk_prevalence_baseline = build_drawdown_risk_prevalence_baseline_comparison(
        risk.predictions,
        supervised,
        risk.fold_metrics,
    )
    opportunity_baseline_challenge = build_opportunity_baseline_challenge(
        outperformance.predictions,
        supervised,
        outperformance.fold_metrics,
        label_col=f"label_outperform_{label_horizon}d",
    )
    (
        opportunity_label_baseline_challenge,
        opportunity_label_baseline_breakdown,
        risk_adjusted_opportunity_fragility,
    ) = build_opportunity_label_baseline_tables(
        target_panel,
        columns,
        horizon=label_horizon,
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    drawdown_risk_feature_group_incremental_value = build_drawdown_risk_feature_group_incremental_value(
        supervised,
        group_options,
        label_column=f"label_drawdown_risk_{label_horizon}d",
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    adverse_outcome_label_comparison = build_adverse_outcome_label_comparison(
        supervised,
        columns,
        current_label_column=f"label_drawdown_risk_{label_horizon}d",
        horizon=label_horizon,
        model_name=model_name,
        train_window=config.train_window,
        test_window=config.test_window,
        step=config.step,
        embargo=config.embargo,
        probability_threshold=config.classification_threshold,
        model_selection_mode=config.model_mode,
    )
    validation_overfit_warnings = build_validation_overfit_warnings(
        outperformance.predictions,
        baseline_panel=supervised,
        universe=",".join(tickers),
    )
    portfolio_correlation, portfolio_crowding, factor_exposure, factor_crowding = (
        build_portfolio_crowding_diagnostics(frames, tickers)
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
        "ml_reliability_by_regime": diagnostics.ml_reliability_by_regime,
        "ml_score_regime_bucket_audit": diagnostics.ml_score_regime_bucket_audit,
        "ml_reliability_gate_diagnostics": diagnostics.ml_reliability_gate_diagnostics,
        "ml_reliability_gate_by_regime": diagnostics.ml_reliability_gate_by_regime,
        "momentum_quality_diagnostics": diagnostics.momentum_quality_diagnostics,
        "momentum_quality_by_regime": diagnostics.momentum_quality_by_regime,
        "momentum_quality_feature_summary": diagnostics.momentum_quality_feature_summary,
        "validation_leakage_diagnostics": validation_leakage,
        "validation_fold_stability": validation_fold_stability,
        "validation_overfit_warnings": validation_overfit_warnings,
        "portfolio_correlation_diagnostics": portfolio_correlation,
        "portfolio_crowding_summary": portfolio_crowding,
        "portfolio_factor_proxy_exposure": factor_exposure,
        "portfolio_factor_crowding_summary": factor_crowding,
        "earnings_event_diagnostics": diagnostics.earnings_event_diagnostics,
        "ml_score_by_earnings_window": diagnostics.ml_score_by_earnings_window,
        "earnings_pead_summary": diagnostics.earnings_pead_summary,
        "drawdown_risk_calibration": diagnostics.drawdown_risk_calibration,
        "drawdown_risk_calibration_quality": diagnostics.drawdown_risk_calibration_quality,
        "drawdown_risk_regime_calibration": diagnostics.drawdown_risk_regime_calibration,
        "drawdown_risk_prevalence_baseline_comparison": drawdown_risk_prevalence_baseline,
        "opportunity_baseline_challenge": opportunity_baseline_challenge,
        "opportunity_label_baseline_challenge": opportunity_label_baseline_challenge,
        "opportunity_label_baseline_breakdown": opportunity_label_baseline_breakdown,
        "risk_adjusted_opportunity_fragility": risk_adjusted_opportunity_fragility,
        "drawdown_risk_feature_group_incremental_value": drawdown_risk_feature_group_incremental_value,
        "adverse_outcome_label_comparison": adverse_outcome_label_comparison,
        "model_selection_summary": pd.concat(
            [
                summarize_model_selection(outperformance.fold_metrics, "outperformance"),
                summarize_model_selection(risk.fold_metrics, "drawdown_risk"),
            ],
            ignore_index=True,
        ),
        "model_selection_fold_details": fold_details,
        "feature_group_comparison": comparison,
        "feature_audit_summary": feature_audit.inventory_summary,
        "feature_family_summary": feature_audit.family_summary,
        "feature_redundancy_selection": feature_audit.redundancy_selection_summary,
        "feature_redundancy_dropped_features": feature_audit.redundancy_selection_report,
        "feature_signal_summary": feature_signal.signal_table,
        "feature_importance_stability": feature_importance_stability,
        "feature_family_importance_stability": feature_family_importance_stability,
        "feature_importance_production_readiness": feature_importance_production_readiness,
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
        "target_stop_rule_comparison": target_stop_rule,
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


def build_opportunity_baseline_challenge(
    predictions: pd.DataFrame,
    baseline_panel: pd.DataFrame,
    fold_details: pd.DataFrame,
    *,
    label_col: str = "label_outperform_20d",
    probability_col: str = "probability",
    bucket_count: int = 4,
    min_train_samples: int = 20,
    min_train_events: int = 5,
) -> pd.DataFrame:
    """Compare opportunity probabilities with train-fold prevalence baselines."""

    required_predictions = {"fold", "Date", "Ticker", "actual", probability_col}
    required_panel = {"Date", "Ticker", label_col}
    required_folds = {"fold", "train_start", "train_end"}
    if (
        predictions.empty
        or baseline_panel.empty
        or fold_details.empty
        or not required_predictions.issubset(predictions.columns)
        or not required_panel.issubset(baseline_panel.columns)
        or not required_folds.issubset(fold_details.columns)
    ):
        return pd.DataFrame(columns=OPPORTUNITY_BASELINE_CHALLENGE_COLUMNS)

    momentum_feature = _select_momentum_feature(baseline_panel)
    regime_col = _select_regime_column(baseline_panel)
    scored = _opportunity_baseline_scored_rows(
        predictions,
        baseline_panel,
        fold_details,
        label_col=label_col,
        probability_col=probability_col,
        regime_col=regime_col,
        momentum_feature=momentum_feature,
        bucket_count=bucket_count,
        min_train_samples=min_train_samples,
        min_train_events=min_train_events,
    )
    if scored.empty:
        return pd.DataFrame(columns=OPPORTUNITY_BASELINE_CHALLENGE_COLUMNS)

    return _opportunity_baseline_summary(scored, probability_col, regime_col, momentum_feature, bucket_count)


def _opportunity_baseline_summary(
    scored: pd.DataFrame,
    probability_col: str,
    regime_col: str | None,
    momentum_feature: str | None,
    bucket_count: int,
) -> pd.DataFrame:
    fold_details_text = ";".join(
        f"{int(row.fold)}:{int(row.train_rows)}:{float(row.global_prevalence):.6f}"
        for row in scored[["fold", "train_rows", "global_prevalence"]].drop_duplicates().itertuples()
    )
    comparators = [
        ("model_predicted_opportunity", probability_col, 0, "model output"),
        ("global_fold_prevalence_baseline", "global_fold_prevalence", 0, "training fold prevalence"),
        (
            "regime_fold_prevalence_baseline",
            "regime_fold_prevalence",
            int(scored["regime_fallback"].sum()),
            "fallback to global fold prevalence",
        ),
        (
            "momentum_bucket_prevalence_baseline",
            "momentum_bucket_prevalence",
            int(scored["momentum_fallback"].sum()),
            "training-fold momentum quantile buckets; fallback to global fold prevalence",
        ),
        (
            "regime_momentum_bucket_prevalence_baseline",
            "regime_momentum_bucket_prevalence",
            int(scored["regime_momentum_fallback"].sum()),
            "training-fold regime+momentum buckets; fallback to regime, then global",
        ),
    ]
    rows = [
        _opportunity_baseline_metrics(
            scored,
            comparator=name,
            probability_col=column,
            regime_col=regime_col,
            fallback_count=fallbacks,
            momentum_feature=momentum_feature,
            bucket_count=bucket_count,
            fallback_rule=fallback_rule,
            fold_details=fold_details_text,
        )
        for name, column, fallbacks, fallback_rule in comparators
    ]
    by_name = {row["comparator"]: row for row in rows}
    model = by_name["model_predicted_opportunity"]
    baseline_results = []
    for name, _, _, _ in comparators[1:]:
        result = _opportunity_baseline_classification(model, by_name[name])
        by_name[name]["classification"] = result
        baseline_results.append(result)
    model["classification"] = _opportunity_model_summary_classification(baseline_results)
    for row in rows:
        row.pop("_worst_fold_brier", None)
        row.pop("_worst_regime_brier", None)
        row.pop("_worst_ticker_brier", None)
    return pd.DataFrame(rows, columns=OPPORTUNITY_BASELINE_CHALLENGE_COLUMNS)


def _opportunity_baseline_breakdown(
    scored: pd.DataFrame,
    probability_col: str,
    regime_col: str | None,
    momentum_feature: str | None,
    bucket_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("fold", "fold"), ("ticker", "Ticker")]
    if regime_col:
        groups.append(("regime", regime_col))

    for breakdown, column in groups:
        if column not in scored:
            continue
        for segment, group in scored.groupby(column, dropna=True, sort=True):
            summary = _opportunity_baseline_summary(group, probability_col, regime_col, momentum_feature, bucket_count)
            for row in summary.to_dict("records"):
                rows.append({"breakdown": breakdown, "segment": segment, **row})

    return pd.DataFrame(
        rows,
        columns=["breakdown", "segment", *OPPORTUNITY_BASELINE_CHALLENGE_COLUMNS],
    )


def build_opportunity_label_baseline_challenge(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    *,
    horizon: int,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    probability_threshold: float = 0.5,
    model_selection_mode: str = "current_default",
    bucket_count: int = 4,
    min_train_samples: int = 20,
    min_train_events: int = 5,
) -> pd.DataFrame:
    """Compare fixed opportunity labels with fold-aware simple baselines."""

    challenge, _, _ = build_opportunity_label_baseline_tables(
        dataset,
        feature_columns,
        horizon=horizon,
        model_name=model_name,
        train_window=train_window,
        test_window=test_window,
        step=step,
        embargo=embargo,
        probability_threshold=probability_threshold,
        model_selection_mode=model_selection_mode,
        bucket_count=bucket_count,
        min_train_samples=min_train_samples,
        min_train_events=min_train_events,
    )
    return challenge


def build_opportunity_label_baseline_tables(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    *,
    horizon: int,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    probability_threshold: float = 0.5,
    model_selection_mode: str = "current_default",
    bucket_count: int = 4,
    min_train_samples: int = 20,
    min_train_events: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare fixed labels with simple baselines, plus local failure breakdowns."""

    if dataset.empty or not feature_columns:
        return (
            pd.DataFrame(columns=OPPORTUNITY_LABEL_BASELINE_CHALLENGE_COLUMNS),
            pd.DataFrame(columns=OPPORTUNITY_LABEL_BASELINE_BREAKDOWN_COLUMNS),
            pd.DataFrame(columns=RISK_ADJUSTED_OPPORTUNITY_FRAGILITY_COLUMNS),
        )

    panel, candidates = _with_opportunity_label_candidates(dataset, horizon)
    rows: list[dict[str, object]] = []
    breakdown_rows: list[dict[str, object]] = []
    fragility_rows: list[dict[str, object]] = []
    for candidate in candidates:
        label_column = candidate["label_column"]
        if label_column not in panel:
            continue
        result = walk_forward_validate_classifier(
            panel,
            feature_columns,
            label_column=label_column,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=probability_threshold,
            model_selection_mode=model_selection_mode,
        )
        momentum_feature = _select_momentum_feature(panel)
        regime_col = _select_regime_column(panel)
        scored = _opportunity_baseline_scored_rows(
            result.predictions,
            panel,
            result.fold_metrics,
            label_col=label_column,
            probability_col="probability",
            regime_col=regime_col,
            momentum_feature=momentum_feature,
            bucket_count=bucket_count,
            min_train_samples=min_train_samples,
            min_train_events=min_train_events,
        )
        if scored.empty:
            continue
        comparison = _opportunity_baseline_summary(scored, "probability", regime_col, momentum_feature, bucket_count)
        for row in comparison.to_dict("records"):
            rows.append({**candidate, **row})
        breakdown = _opportunity_baseline_breakdown(scored, "probability", regime_col, momentum_feature, bucket_count)
        for row in breakdown.to_dict("records"):
            breakdown_rows.append({**candidate, **row})
        if candidate["target_id"] == f"risk_adjusted_excess_{horizon}d":
            fragility = _risk_adjusted_opportunity_fragility(scored, candidate, regime_col, momentum_feature, bucket_count)
            fragility_rows.extend(fragility.to_dict("records"))
    return (
        pd.DataFrame(rows, columns=OPPORTUNITY_LABEL_BASELINE_CHALLENGE_COLUMNS),
        pd.DataFrame(breakdown_rows, columns=OPPORTUNITY_LABEL_BASELINE_BREAKDOWN_COLUMNS),
        pd.DataFrame(fragility_rows, columns=RISK_ADJUSTED_OPPORTUNITY_FRAGILITY_COLUMNS),
    )


def _risk_adjusted_opportunity_fragility(
    scored: pd.DataFrame,
    candidate: dict[str, str],
    regime_col: str | None,
    momentum_feature: str | None,
    bucket_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ticker_rows: list[dict[str, object]] = []
    for ticker, group in scored.groupby("Ticker", dropna=True, sort=True):
        row = _risk_adjusted_fragility_row(
            group, candidate, "ticker", str(ticker), "", regime_col, momentum_feature, bucket_count
        )
        ticker_rows.append(row)
        rows.append(row)

    for fold, group in scored.groupby("fold", dropna=True, sort=True):
        rows.append(
            _risk_adjusted_fragility_row(
                group, candidate, "fold_ticker_mix", str(fold), "", regime_col, momentum_feature, bucket_count
            )
        )

    if regime_col and regime_col in scored:
        for regime, group in scored.groupby(regime_col, dropna=True, sort=True):
            rows.append(
                _risk_adjusted_fragility_row(
                    group, candidate, "regime_ticker_mix", str(regime), "", regime_col, momentum_feature, bucket_count
                )
            )
        high_vol = scored[scored[regime_col].astype(str).str.lower().eq("uptrend / high volatility")]
        for ticker, group in high_vol.groupby("Ticker", dropna=True, sort=True):
            rows.append(
                _risk_adjusted_fragility_row(
                    group,
                    candidate,
                    "high_vol_uptrend_ticker",
                    str(ticker),
                    "",
                    regime_col,
                    momentum_feature,
                    bucket_count,
                )
            )
        for excluded, view in (
            (["PLTR"], "high_vol_uptrend_exclude_pltr"),
            (["PLTR", "TSLA"], "high_vol_uptrend_exclude_pltr_tsla"),
        ):
            remaining = high_vol[~high_vol["Ticker"].astype(str).isin(excluded)]
            if not remaining.empty:
                rows.append(
                    _risk_adjusted_fragility_row(
                        remaining,
                        candidate,
                        view,
                        "all_remaining",
                        ",".join(excluded),
                        regime_col,
                        momentum_feature,
                        bucket_count,
                    )
                )

    worst = [
        str(row["segment"])
        for row in sorted(
            ticker_rows,
            key=lambda row: (
                -int(row["baseline_loss_count"]),
                -float(pd.to_numeric(pd.Series([row["model_brier_score"]]), errors="coerce").fillna(-1).iloc[0]),
                str(row["segment"]),
            ),
        )
    ]
    for count, view in ((1, "exclude_worst_ticker"), (2, "exclude_worst_two_tickers")):
        excluded = worst[:count]
        if not excluded:
            continue
        remaining = scored[~scored["Ticker"].astype(str).isin(excluded)]
        rows.append(
            _risk_adjusted_fragility_row(
                remaining,
                candidate,
                view,
                "all_remaining",
                ",".join(excluded),
                regime_col,
                momentum_feature,
                bucket_count,
            )
        )

    return pd.DataFrame(rows, columns=RISK_ADJUSTED_OPPORTUNITY_FRAGILITY_COLUMNS)


def _risk_adjusted_fragility_row(
    group: pd.DataFrame,
    candidate: dict[str, str],
    view: str,
    segment: str,
    excluded_tickers: str,
    regime_col: str | None,
    momentum_feature: str | None,
    bucket_count: int,
) -> dict[str, object]:
    comparison = _opportunity_baseline_summary(group, "probability", regime_col, momentum_feature, bucket_count)
    model = comparison[comparison["comparator"] == "model_predicted_opportunity"].iloc[0].to_dict()
    baselines = comparison[comparison["comparator"] != "model_predicted_opportunity"]
    counts = baselines["classification"].value_counts()
    inverted = classification_metrics(group["actual"], 1.0 - group["probability"])
    model_bucket_spread = _edge_bucket_spread(group, "probability", "actual", bucket_count)
    inverted_group = group.assign(_inverted_probability=1.0 - group["probability"])
    inverted_bucket_spread = _edge_bucket_spread(inverted_group, "_inverted_probability", "actual", bucket_count)
    outcome_col = _risk_adjusted_outcome_column(candidate, group)
    normal_outcome = _edge_bucket_outcome(group, "probability", outcome_col, bucket_count)
    inverted_outcome = _edge_bucket_outcome(inverted_group, "_inverted_probability", outcome_col, bucket_count)
    row = {column: pd.NA for column in RISK_ADJUSTED_OPPORTUNITY_FRAGILITY_COLUMNS}
    row.update(
        {
            **candidate,
            "view": view,
            "segment": segment,
            "excluded_tickers": excluded_tickers,
            "sample_count": model["sample_count"],
            "ticker_count": model["ticker_count"],
            "fold_count": model["fold_count"],
            "event_prevalence": model["event_prevalence"],
            "model_roc_auc": model["roc_auc"],
            "model_pr_auc": model["pr_auc"],
            "model_brier_score": model["brier_score"],
            "model_calibration_gap": model["calibration_gap"],
            "model_bucket_spread": model_bucket_spread,
            "inverted_roc_auc": inverted.get("roc_auc", pd.NA),
            "inverted_pr_auc": inverted.get("pr_auc", pd.NA),
            "inverted_brier_score": inverted.get("brier_score", pd.NA),
            "inverted_calibration_gap": float(group["actual"].mean() - (1.0 - group["probability"]).mean())
            if not group.empty
            else pd.NA,
            "inverted_bucket_spread": inverted_bucket_spread,
            "outcome_column": outcome_col or pd.NA,
            "normal_top_bucket_forward_outcome": normal_outcome["top"],
            "normal_bottom_bucket_forward_outcome": normal_outcome["bottom"],
            "normal_forward_outcome_spread": normal_outcome["spread"],
            "inverted_top_bucket_forward_outcome": inverted_outcome["top"],
            "inverted_bottom_bucket_forward_outcome": inverted_outcome["bottom"],
            "inverted_forward_outcome_spread": inverted_outcome["spread"],
            "baseline_loss_count": int(counts.get("baseline_beats_model", 0)),
            "baseline_match_count": int(counts.get("baseline_matches_model", 0)),
            "model_win_count": int(counts.get("model_beats_baseline", 0)),
            "unstable_count": int(counts.get("unstable_or_inconclusive", 0)),
            "worst_ticker": model["worst_ticker"],
            "worst_fold": model["worst_fold"],
            "worst_regime": model["worst_regime"],
            "ticker_mix": _ticker_mix(group),
            "classification": model["classification"],
        }
    )
    return row


def _edge_bucket_spread(data: pd.DataFrame, score_col: str, value_col: str, bucket_count: int) -> object:
    outcome = _edge_bucket_outcome(data, score_col, value_col, bucket_count)
    return outcome["spread"]


def _edge_bucket_outcome(
    data: pd.DataFrame,
    score_col: str,
    value_col: str | None,
    bucket_count: int,
) -> dict[str, object]:
    empty = {"bottom": pd.NA, "top": pd.NA, "spread": pd.NA}
    if not value_col or score_col not in data or value_col not in data:
        return empty
    clean = data[[score_col, value_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 2 or clean[score_col].nunique() < 2:
        return empty
    buckets = min(max(2, bucket_count), len(clean))
    clean = clean.assign(_bucket=pd.qcut(clean[score_col].rank(method="first"), q=buckets, labels=False))
    grouped = clean.groupby("_bucket", observed=True)[value_col].mean()
    if len(grouped) < 2:
        return empty
    bottom = float(grouped.iloc[0])
    top = float(grouped.iloc[-1])
    return {"bottom": bottom, "top": top, "spread": top - bottom}


def _risk_adjusted_outcome_column(candidate: dict[str, str], data: pd.DataFrame) -> str | None:
    label_column = str(candidate.get("label_column", ""))
    suffix = label_column.rsplit("_", 1)[-1]
    horizon = suffix if suffix.endswith("d") else "20d"
    for column in (
        f"forward_{horizon}_recent_vol_adjusted_excess_return",
        f"forward_{horizon}_excess_return",
        "forward_risk_adjusted_excess_return",
        "forward_excess_return",
    ):
        if column in data and pd.to_numeric(data[column], errors="coerce").notna().any():
            return column
    return None


def _ticker_mix(data: pd.DataFrame) -> str:
    counts = data["Ticker"].astype(str).value_counts().sort_index()
    return ";".join(f"{ticker}:{int(count)}" for ticker, count in counts.items())


def _with_opportunity_label_candidates(dataset: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    output = dataset.copy()
    excess_col = f"forward_{horizon}d_excess_return"
    drawdown_col = f"forward_{horizon}d_drawdown"
    stronger_label = f"label_stronger_excess_{horizon}d"
    composite_label = f"label_composite_opportunity_{horizon}d"

    excess = pd.to_numeric(output.get(excess_col, pd.Series(pd.NA, index=output.index)), errors="coerce")
    drawdown = pd.to_numeric(output.get(drawdown_col, pd.Series(pd.NA, index=output.index)), errors="coerce")
    output[stronger_label] = (excess > 0.05).astype(float)
    output.loc[excess.isna(), stronger_label] = pd.NA
    output[composite_label] = ((excess > 0.0) & (drawdown >= -0.10)).astype(float)
    output.loc[excess.isna() | drawdown.isna(), composite_label] = pd.NA

    candidates = [
        {
            "target_id": f"outperform_{horizon}d",
            "display_name": f"Current {horizon}d outperformance",
            "label_column": f"label_outperform_{horizon}d",
        },
        {
            "target_id": f"stronger_excess_{horizon}d",
            "display_name": f"Stronger {horizon}d excess return",
            "label_column": stronger_label,
        },
        {
            "target_id": f"top_tercile_excess_{horizon}d",
            "display_name": f"Top-third relative performer",
            "label_column": f"label_top_tercile_excess_{horizon}d",
        },
        {
            "target_id": f"risk_adjusted_excess_{horizon}d",
            "display_name": f"Recent-vol adjusted excess",
            "label_column": f"label_risk_adjusted_excess_{horizon}d",
        },
        {
            "target_id": f"composite_opportunity_{horizon}d",
            "display_name": f"Positive excess with acceptable drawdown",
            "label_column": composite_label,
        },
    ]
    return output, candidates


def _select_momentum_feature(panel: pd.DataFrame) -> str | None:
    for column in MOMENTUM_BASELINE_CANDIDATES:
        if column in panel and pd.to_numeric(panel[column], errors="coerce").notna().any():
            return column
    return None


def _select_regime_column(panel: pd.DataFrame) -> str | None:
    for column in DEFAULT_REGIME_COLUMNS:
        if column in panel and panel[column].notna().any():
            return column
    return None


def _opportunity_baseline_scored_rows(
    predictions: pd.DataFrame,
    baseline_panel: pd.DataFrame,
    fold_details: pd.DataFrame,
    *,
    label_col: str,
    probability_col: str,
    regime_col: str | None,
    momentum_feature: str | None,
    bucket_count: int,
    min_train_samples: int,
    min_train_events: int,
) -> pd.DataFrame:
    merge_cols = ["Date", "Ticker"]
    extra_cols = [
        column
        for column in [label_col, regime_col, momentum_feature, *_risk_adjusted_forward_columns(label_col)]
        if column and column in baseline_panel
    ]
    extra_cols = list(dict.fromkeys(extra_cols))
    panel = baseline_panel[[*merge_cols, *extra_cols]].copy()
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel[label_col] = pd.to_numeric(panel[label_col], errors="coerce")
    if regime_col:
        panel[regime_col] = panel[regime_col].astype(str).str.strip()
    if momentum_feature:
        panel[momentum_feature] = pd.to_numeric(panel[momentum_feature], errors="coerce")

    usable = predictions.rename(columns={probability_col: "model_probability"}).copy()
    usable["Date"] = pd.to_datetime(usable["Date"])
    usable["actual"] = pd.to_numeric(usable["actual"], errors="coerce")
    usable["model_probability"] = pd.to_numeric(usable["model_probability"], errors="coerce").clip(0.0, 1.0)
    usable = usable.merge(panel, on=merge_cols, how="left")
    usable = usable.rename(columns={"model_probability": probability_col})
    usable = usable.dropna(subset=["fold", "Date", "Ticker", "actual", probability_col])
    if usable.empty:
        return pd.DataFrame()

    fold_rows = fold_details.drop_duplicates(subset=["fold"], keep="last").set_index("fold")
    frames = []
    for fold, group in usable.groupby("fold", sort=True):
        if fold not in fold_rows.index:
            continue
        fold_row = fold_rows.loc[fold]
        train = panel[
            (panel["Date"] >= pd.to_datetime(fold_row["train_start"]))
            & (panel["Date"] <= pd.to_datetime(fold_row["train_end"]))
            & panel[label_col].notna()
        ].copy()
        if train.empty:
            continue
        global_prevalence = float(train[label_col].mean())
        out = group.copy()
        out["global_prevalence"] = global_prevalence
        out["train_rows"] = len(train)
        out["global_fold_prevalence"] = global_prevalence
        out["regime_fold_prevalence"] = global_prevalence
        out["momentum_bucket_prevalence"] = global_prevalence
        out["regime_momentum_bucket_prevalence"] = global_prevalence
        out["regime_fallback"] = False
        out["momentum_fallback"] = False
        out["regime_momentum_fallback"] = False

        regime_rates = _prevalence_lookup(train, label_col, [regime_col] if regime_col else [], min_train_samples, min_train_events)
        if regime_col:
            for idx, row in out.iterrows():
                key = (str(row.get(regime_col)).strip(),)
                if key in regime_rates:
                    out.at[idx, "regime_fold_prevalence"] = regime_rates[key]
                else:
                    out.at[idx, "regime_fallback"] = True

        if momentum_feature:
            edges = _training_quantile_edges(train[momentum_feature], bucket_count)
            if edges is not None:
                train["_momentum_bucket"] = pd.cut(
                    train[momentum_feature], bins=edges, labels=False, include_lowest=True
                )
                out["_momentum_bucket"] = pd.cut(
                    out[momentum_feature], bins=edges, labels=False, include_lowest=True
                )
                momentum_rates = _prevalence_lookup(
                    train, label_col, ["_momentum_bucket"], min_train_samples, min_train_events
                )
                combo_keys = [regime_col, "_momentum_bucket"] if regime_col else []
                combo_rates = _prevalence_lookup(train, label_col, combo_keys, min_train_samples, min_train_events)
                for idx, row in out.iterrows():
                    bucket_key = (row.get("_momentum_bucket"),)
                    if bucket_key in momentum_rates:
                        out.at[idx, "momentum_bucket_prevalence"] = momentum_rates[bucket_key]
                    else:
                        out.at[idx, "momentum_fallback"] = True

                    combo_key = (str(row.get(regime_col)).strip(), row.get("_momentum_bucket")) if regime_col else ()
                    if combo_key in combo_rates:
                        out.at[idx, "regime_momentum_bucket_prevalence"] = combo_rates[combo_key]
                    else:
                        out.at[idx, "regime_momentum_fallback"] = True
                        out.at[idx, "regime_momentum_bucket_prevalence"] = out.at[idx, "regime_fold_prevalence"]
            else:
                out["momentum_fallback"] = True
                out["regime_momentum_fallback"] = True
        else:
            out["momentum_fallback"] = True
            out["regime_momentum_fallback"] = True
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _risk_adjusted_forward_columns(label_col: str) -> list[str]:
    suffix = label_col.rsplit("_", 1)[-1]
    horizon = suffix if suffix.endswith("d") else "20d"
    return [
        f"forward_{horizon}_recent_vol_adjusted_excess_return",
        f"forward_{horizon}_excess_return",
    ]


def _training_quantile_edges(values: pd.Series, bucket_count: int) -> list[float] | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < bucket_count * 2 or clean.nunique() < bucket_count:
        return None
    quantiles = clean.quantile([i / bucket_count for i in range(bucket_count + 1)]).drop_duplicates().to_list()
    if len(quantiles) < 3:
        return None
    quantiles[0] = float("-inf")
    quantiles[-1] = float("inf")
    return quantiles


def _prevalence_lookup(
    data: pd.DataFrame,
    label_col: str,
    keys: list[str],
    min_train_samples: int,
    min_train_events: int,
) -> dict[tuple[object, ...], float]:
    if not keys:
        return {}
    rates = {}
    for key, group in data.dropna(subset=keys).groupby(keys, dropna=True):
        event_count = int((group[label_col] == 1).sum())
        non_event_count = int((group[label_col] == 0).sum())
        if len(group) >= min_train_samples and event_count >= min_train_events and non_event_count >= min_train_events:
            rates[key if isinstance(key, tuple) else (key,)] = float(group[label_col].mean())
    return rates


def _opportunity_baseline_metrics(
    data: pd.DataFrame,
    *,
    comparator: str,
    probability_col: str,
    regime_col: str | None,
    fallback_count: int,
    momentum_feature: str | None,
    bucket_count: int,
    fallback_rule: str,
    fold_details: str,
) -> dict[str, object]:
    metrics = classification_metrics(data["actual"], data[probability_col])
    event_prevalence = float(data["actual"].mean()) if not data.empty else pd.NA
    mean_predicted = float(data[probability_col].mean()) if not data.empty else pd.NA
    return {
        "comparator": comparator,
        "sample_count": int(len(data)),
        "ticker_count": int(data["Ticker"].nunique()) if "Ticker" in data else 0,
        "fold_count": int(data["fold"].nunique()) if "fold" in data else 0,
        "event_prevalence": event_prevalence,
        "mean_predicted_opportunity": mean_predicted,
        "roc_auc": metrics.get("roc_auc", pd.NA),
        "pr_auc": metrics.get("pr_auc", pd.NA),
        "brier_score": metrics.get("brier_score", pd.NA),
        "calibration_gap": event_prevalence - mean_predicted
        if pd.notna(event_prevalence) and pd.notna(mean_predicted)
        else pd.NA,
        "worst_regime": _worst_brier_group(data, probability_col, regime_col) if regime_col else pd.NA,
        "worst_fold": _worst_brier_group(data, probability_col, "fold"),
        "worst_ticker": _worst_brier_group(data, probability_col, "Ticker"),
        "fallback_count": fallback_count,
        "momentum_feature": momentum_feature or pd.NA,
        "bucket_count": bucket_count if momentum_feature else pd.NA,
        "fallback_rule": fallback_rule,
        "fold_train_prevalence_details": fold_details,
        "classification": "insufficient_evidence",
        "_worst_fold_brier": _worst_brier(data, probability_col, "fold"),
        "_worst_regime_brier": _worst_brier(data, probability_col, regime_col) if regime_col else pd.NA,
        "_worst_ticker_brier": _worst_brier(data, probability_col, "Ticker"),
    }


def _worst_brier_group(data: pd.DataFrame, probability_col: str, group_col: str | None) -> object:
    if not group_col or group_col not in data:
        return pd.NA
    values = [
        (value, float(((group[probability_col] - group["actual"]) ** 2).mean()))
        for value, group in data.groupby(group_col, dropna=True, sort=True)
        if not group.empty
    ]
    return pd.NA if not values else max(values, key=lambda item: item[1])[0]


def _worst_brier(data: pd.DataFrame, probability_col: str, group_col: str | None) -> object:
    if not group_col or group_col not in data:
        return pd.NA
    values = [
        float(((group[probability_col] - group["actual"]) ** 2).mean())
        for _, group in data.groupby(group_col, dropna=True)
        if not group.empty
    ]
    return pd.NA if not values else max(values)


def _opportunity_baseline_classification(model: dict[str, object], baseline: dict[str, object]) -> str:
    values = {
        name: pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        for name, row, column in (
            ("model_pr", model, "pr_auc"),
            ("baseline_pr", baseline, "pr_auc"),
            ("model_brier", model, "brier_score"),
            ("baseline_brier", baseline, "brier_score"),
            ("model_gap", model, "calibration_gap"),
            ("baseline_gap", baseline, "calibration_gap"),
            ("model_worst_fold", model, "_worst_fold_brier"),
            ("baseline_worst_fold", baseline, "_worst_fold_brier"),
            ("model_worst_regime", model, "_worst_regime_brier"),
            ("baseline_worst_regime", baseline, "_worst_regime_brier"),
        )
    }
    if any(pd.isna(value) for value in values.values()):
        return "insufficient_evidence"

    model_wins = sum(
        (
            values["model_pr"] > values["baseline_pr"] + 0.01,
            values["model_brier"] + 0.0025 < values["baseline_brier"],
            abs(values["model_gap"]) + 0.025 < abs(values["baseline_gap"]),
            values["model_worst_fold"] + 0.01 < values["baseline_worst_fold"],
            values["model_worst_regime"] + 0.01 < values["baseline_worst_regime"],
        )
    )
    baseline_wins = sum(
        (
            values["baseline_pr"] > values["model_pr"] + 0.01,
            values["baseline_brier"] + 0.0025 < values["model_brier"],
            abs(values["baseline_gap"]) + 0.025 < abs(values["model_gap"]),
            values["baseline_worst_fold"] + 0.01 < values["model_worst_fold"],
            values["baseline_worst_regime"] + 0.01 < values["model_worst_regime"],
        )
    )
    if model_wins >= 3 and baseline_wins <= 1:
        return "model_beats_baseline"
    if baseline_wins >= 3 and model_wins <= 1:
        return "baseline_beats_model"
    if max(model_wins, baseline_wins) <= 2:
        return "baseline_matches_model"
    return "unstable_or_inconclusive"


def _opportunity_model_summary_classification(baseline_results: list[str]) -> str:
    if not baseline_results or all(result == "insufficient_evidence" for result in baseline_results):
        return "insufficient_evidence"
    if "baseline_beats_model" in baseline_results:
        return "baseline_beats_model"
    if "unstable_or_inconclusive" in baseline_results:
        return "unstable_or_inconclusive"
    if all(result == "model_beats_baseline" for result in baseline_results):
        return "model_beats_baseline"
    return "baseline_matches_model"


def build_drawdown_risk_feature_group_incremental_value(
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    label_column: str,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    probability_threshold: float = 0.5,
    model_selection_mode: str = "current_default",
) -> pd.DataFrame:
    """Compare drawdown-risk feature groups with fold-aware prevalence baselines."""

    rows: list[dict[str, object]] = []
    for feature_group, columns in feature_groups.items():
        if not columns:
            continue
        result = walk_forward_validate_classifier(
            dataset,
            columns,
            label_column=label_column,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=probability_threshold,
            model_selection_mode=model_selection_mode,
        )
        comparison = build_drawdown_risk_prevalence_baseline_comparison(
            result.predictions,
            dataset,
            result.fold_metrics,
            label_col=label_column,
        )
        if comparison.empty:
            rows.append(_insufficient_risk_incremental_row(feature_group, len(columns)))
            continue
        rows.append(_risk_incremental_row(feature_group, len(columns), comparison))
    return pd.DataFrame(rows, columns=RISK_INCREMENTAL_VALUE_COLUMNS)


def build_adverse_outcome_label_comparison(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    *,
    current_label_column: str,
    horizon: int,
    model_name: str,
    train_window: int,
    test_window: int,
    step: int | None,
    embargo: int,
    probability_threshold: float = 0.5,
    model_selection_mode: str = "current_default",
) -> pd.DataFrame:
    """Compare fixed research-only adverse labels with prevalence baselines."""

    panel, candidates = _with_adverse_outcome_labels(dataset, horizon)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        label = str(candidate["label"])
        result = walk_forward_validate_classifier(
            panel,
            feature_columns,
            label_column=label,
            model_name=model_name,
            train_window=train_window,
            test_window=test_window,
            step=step,
            embargo=embargo,
            probability_threshold=probability_threshold,
            model_selection_mode=model_selection_mode,
        )
        comparison = build_drawdown_risk_prevalence_baseline_comparison(
            result.predictions,
            panel,
            result.fold_metrics,
            label_col=label,
        )
        rows.append(_adverse_outcome_label_row(panel, current_label_column, candidate, comparison))
    return pd.DataFrame(rows, columns=ADVERSE_OUTCOME_LABEL_COMPARISON_COLUMNS)


def _with_adverse_outcome_labels(dataset: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    output = dataset.copy()
    drawdown_col = f"forward_{horizon}d_drawdown"
    excess_col = f"forward_{horizon}d_excess_return"
    return_col = f"forward_{horizon}d_return"
    candidates = [
        {
            "label": f"risk_severe_drawdown_{horizon}d",
            "definition": f"future {horizon}d max drawdown < -15%",
            "threshold": "forward_drawdown < -0.15",
        },
        {
            "label": f"risk_negative_excess_{horizon}d",
            "definition": f"future {horizon}d excess return vs benchmark < -5%",
            "threshold": "forward_excess_return < -0.05",
        },
        {
            "label": f"risk_composite_drawdown_underperform_{horizon}d",
            "definition": f"future {horizon}d drawdown < -10% and excess return < -2%",
            "threshold": "forward_drawdown < -0.10 and forward_excess_return < -0.02",
        },
    ]
    output[candidates[0]["label"]] = (_numeric(output.get(drawdown_col)) < -0.15).astype(float)
    output[candidates[1]["label"]] = (_numeric(output.get(excess_col)) < -0.05).astype(float)
    output[candidates[2]["label"]] = (
        (_numeric(output.get(drawdown_col)) < -0.10) & (_numeric(output.get(excess_col)) < -0.02)
    ).astype(float)
    output.loc[_numeric(output.get(drawdown_col)).isna(), candidates[0]["label"]] = pd.NA
    output.loc[_numeric(output.get(excess_col)).isna(), candidates[1]["label"]] = pd.NA
    output.loc[
        _numeric(output.get(drawdown_col)).isna() | _numeric(output.get(excess_col)).isna(),
        candidates[2]["label"],
    ] = pd.NA

    if "volatility_20d" in output and return_col in output and drawdown_col in output:
        vol_label = f"risk_vol_adjusted_adverse_{horizon}d"
        vol_move = _numeric(output["volatility_20d"]) * (horizon / 252) ** 0.5
        output[vol_label] = (
            (_numeric(output[return_col]) < -vol_move) | (_numeric(output[drawdown_col]) < -vol_move)
        ).astype(float)
        output.loc[
            vol_move.isna() | _numeric(output[return_col]).isna() | _numeric(output[drawdown_col]).isna(),
            vol_label,
        ] = pd.NA
        candidates.append(
            {
                "label": vol_label,
                "definition": f"future {horizon}d return or drawdown worse than trailing-volatility move",
                "threshold": "forward_return or forward_drawdown < -volatility_20d*sqrt(20/252)",
            }
        )
    return output, candidates


def _adverse_outcome_label_row(
    panel: pd.DataFrame,
    current_label_column: str,
    candidate: dict[str, str],
    comparison: pd.DataFrame,
) -> dict[str, object]:
    label = candidate["label"]
    row = {column: pd.NA for column in ADVERSE_OUTCOME_LABEL_COMPARISON_COLUMNS}
    row.update({key: candidate[key] for key in ("label", "definition", "threshold")})
    if comparison.empty:
        row["classification"] = "insufficient_evidence"
        return row

    by_name = {str(item["comparator"]): item for _, item in comparison.iterrows()}
    model = by_name.get("model_predicted_risk")
    global_base = by_name.get("global_fold_prevalence_baseline")
    regime_base = by_name.get("regime_fold_prevalence_baseline")
    if model is None or global_base is None or regime_base is None:
        row["classification"] = "insufficient_evidence"
        return row

    row.update(
        {
            "sample_count": model["sample_count"],
            "ticker_count": model["ticker_count"],
            "fold_count": model["fold_count"],
            "event_prevalence": model["event_prevalence"],
            "current_label_overlap_rate": _current_label_overlap(panel, label, current_label_column),
            **_regime_concentration(panel, label),
            "model_roc_auc": model["roc_auc"],
            "model_pr_auc": model["pr_auc"],
            "model_brier_score": model["brier_score"],
            "model_calibration_gap": model["calibration_gap"],
            "global_fold_baseline_brier_score": global_base["brier_score"],
            "global_fold_baseline_calibration_gap": global_base["calibration_gap"],
            "regime_fold_baseline_brier_score": regime_base["brier_score"],
            "regime_fold_baseline_calibration_gap": regime_base["calibration_gap"],
            "model_vs_global_fold_baseline": global_base["classification"],
            "model_vs_regime_fold_baseline": regime_base["classification"],
            "worst_regime": model["worst_regime"],
            "worst_fold": model["worst_fold"],
            "worst_ticker": model["worst_ticker"],
        }
    )
    row["classification"] = _adverse_label_classification(row)
    return row


def _numeric(values: object) -> pd.Series:
    return pd.to_numeric(values if values is not None else pd.Series(dtype=float), errors="coerce")


def _current_label_overlap(panel: pd.DataFrame, label: str, current_label_column: str) -> object:
    if label not in panel or current_label_column not in panel:
        return pd.NA
    data = panel[[label, current_label_column]].apply(pd.to_numeric, errors="coerce").dropna()
    positives = data[data[label] == 1]
    return pd.NA if positives.empty else float((positives[current_label_column] == 1).mean())


def _regime_concentration(panel: pd.DataFrame, label: str) -> dict[str, object]:
    regime_col = next((column for column in ("market_regime", "regime") if column in panel), None)
    if regime_col is None or label not in panel:
        return {"regime_concentration": pd.NA, "most_concentrated_regime": pd.NA}
    positives = panel[pd.to_numeric(panel[label], errors="coerce") == 1]
    if positives.empty:
        return {"regime_concentration": pd.NA, "most_concentrated_regime": pd.NA}
    shares = positives[regime_col].astype(str).value_counts(normalize=True)
    return {"regime_concentration": float(shares.iloc[0]), "most_concentrated_regime": shares.index[0]}


def _adverse_label_classification(row: dict[str, object]) -> str:
    results = {str(row.get("model_vs_global_fold_baseline")), str(row.get("model_vs_regime_fold_baseline"))}
    if "insufficient_evidence" in results:
        return "insufficient_evidence"
    if _rare_event_cosmetic_brier_only(row):
        return "rare_event_cosmetic_brier_only"
    if results == {"model_beats_baseline"}:
        return "candidate_beats_baseline"
    if "baseline_beats_model" in results:
        return "baseline_beats_candidate"
    if results == {"baseline_matches_model"}:
        return "baseline_matches_candidate"
    return "unstable_or_inconclusive"


def _rare_event_cosmetic_brier_only(row: dict[str, object]) -> bool:
    prevalence = pd.to_numeric(pd.Series([row.get("event_prevalence")]), errors="coerce").iloc[0]
    pr_auc = pd.to_numeric(pd.Series([row.get("model_pr_auc")]), errors="coerce").iloc[0]
    brier = pd.to_numeric(pd.Series([row.get("model_brier_score")]), errors="coerce").iloc[0]
    global_brier = pd.to_numeric(pd.Series([row.get("global_fold_baseline_brier_score")]), errors="coerce").iloc[0]
    regime_brier = pd.to_numeric(pd.Series([row.get("regime_fold_baseline_brier_score")]), errors="coerce").iloc[0]
    concentration = pd.to_numeric(pd.Series([row.get("regime_concentration")]), errors="coerce").iloc[0]
    if pd.isna(prevalence) or prevalence >= 0.05 or pd.isna(brier):
        return False
    brier_not_worse = (pd.isna(global_brier) or brier <= global_brier) and (
        pd.isna(regime_brier) or brier <= regime_brier
    )
    weak_pr = pd.isna(pr_auc) or pr_auc <= max(0.05, prevalence * 1.5)
    concentrated = pd.notna(concentration) and concentration >= 0.75
    return bool(brier_not_worse and (weak_pr or concentrated))


def _insufficient_risk_incremental_row(feature_group: str, features: int) -> dict[str, object]:
    row = {column: pd.NA for column in RISK_INCREMENTAL_VALUE_COLUMNS}
    row.update({"feature_group": feature_group, "features": features, "classification": "insufficient_evidence"})
    return row


def _risk_incremental_row(feature_group: str, features: int, comparison: pd.DataFrame) -> dict[str, object]:
    by_name = {str(row["comparator"]): row for _, row in comparison.iterrows()}
    model = by_name.get("model_predicted_risk")
    global_base = by_name.get("global_fold_prevalence_baseline")
    regime_base = by_name.get("regime_fold_prevalence_baseline")
    if model is None or global_base is None or regime_base is None:
        return _insufficient_risk_incremental_row(feature_group, features)

    global_result = _risk_baseline_result(model, global_base)
    regime_result = _risk_baseline_result(model, regime_base)
    return {
        "feature_group": feature_group,
        "features": features,
        "sample_count": model["sample_count"],
        "ticker_count": model["ticker_count"],
        "fold_count": model["fold_count"],
        "event_prevalence": model["event_prevalence"],
        "model_roc_auc": model["roc_auc"],
        "model_pr_auc": model["pr_auc"],
        "model_brier_score": model["brier_score"],
        "model_calibration_gap": model["calibration_gap"],
        "global_fold_baseline_roc_auc": global_base["roc_auc"],
        "global_fold_baseline_pr_auc": global_base["pr_auc"],
        "global_fold_baseline_brier_score": global_base["brier_score"],
        "global_fold_baseline_calibration_gap": global_base["calibration_gap"],
        "regime_fold_baseline_roc_auc": regime_base["roc_auc"],
        "regime_fold_baseline_pr_auc": regime_base["pr_auc"],
        "regime_fold_baseline_brier_score": regime_base["brier_score"],
        "regime_fold_baseline_calibration_gap": regime_base["calibration_gap"],
        "model_vs_global_fold_baseline": global_result,
        "model_vs_regime_fold_baseline": regime_result,
        "worst_regime": model["worst_regime"],
        "worst_fold": model["worst_fold"],
        "worst_ticker": model["worst_ticker"],
        "fallback_count": regime_base["fallback_count"],
        "fold_train_prevalence_details": model["fold_train_prevalence_details"],
        "classification": _risk_incremental_classification(global_result, regime_result),
    }


def _risk_baseline_result(model: pd.Series, baseline: pd.Series) -> str:
    model_status = str(model.get("classification", "insufficient_evidence"))
    baseline_status = str(baseline.get("classification", "insufficient_evidence"))
    if "insufficient_evidence" in {model_status, baseline_status}:
        return "insufficient_evidence"
    if baseline_status == "baseline_beats_model":
        return "baseline_beats_feature_group"
    if baseline_status == "model_beats_baseline":
        return "adds_incremental_risk_signal"
    return "baseline_matches_feature_group"


def _risk_incremental_classification(global_result: str, regime_result: str) -> str:
    results = {global_result, regime_result}
    if "insufficient_evidence" in results:
        return "insufficient_evidence"
    if "baseline_beats_feature_group" in results:
        return "baseline_beats_feature_group"
    if results == {"adds_incremental_risk_signal"}:
        return "adds_incremental_risk_signal"
    if results == {"baseline_matches_feature_group"}:
        return "baseline_matches_feature_group"
    return "unstable_or_inconclusive"


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
