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
