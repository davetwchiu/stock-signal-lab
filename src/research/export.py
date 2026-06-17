"""Local Research Lab diagnostics bundle export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


SAFE_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class ResearchExportResult:
    """Filesystem locations and manifest for a completed export."""

    run_dir: Path
    latest_dir: Path
    manifest: dict[str, object]
    codex_handoff: str


def build_research_lab_export_payload(
    *,
    run_metadata: Mapping[str, object],
    tables: Mapping[str, object],
) -> dict[str, object]:
    """Build the app-facing payload exported after Streamlit reruns."""

    return {
        "run_metadata": dict(run_metadata),
        "tables": dict(tables),
    }


def export_research_lab_payload(
    payload: Mapping[str, object],
    *,
    output_root: str | Path = "data/research_runs",
    notes: Mapping[str, str] | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> ResearchExportResult:
    """Export the same payload stored by the Research Lab UI."""

    run_metadata = payload.get("run_metadata", {})
    tables = payload.get("tables", {})
    if not isinstance(run_metadata, Mapping):
        raise TypeError("Research export payload run_metadata must be a mapping")
    if not isinstance(tables, Mapping):
        raise TypeError("Research export payload tables must be a mapping")
    return export_research_lab_diagnostics(
        run_metadata=run_metadata,
        tables=tables,
        output_root=output_root,
        notes=notes,
        run_id=run_id,
        created_at=created_at,
    )


def current_git_commit() -> str | None:
    """Return the current git commit if it is cheaply available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def export_research_lab_diagnostics(
    *,
    run_metadata: Mapping[str, object],
    tables: Mapping[str, object],
    output_root: str | Path = "data/research_runs",
    notes: Mapping[str, str] | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> ResearchExportResult:
    """Write a local diagnostics bundle from already-computed Research Lab tables."""

    export_time = created_at or datetime.now().replace(microsecond=0)
    created_at_text = export_time.isoformat()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    folder_name = run_id or export_time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = _unique_run_dir(root / folder_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = dict(run_metadata)
    metadata.setdefault("created_at", created_at_text)
    metadata.setdefault("app_name", "Stock Signal Lab")
    metadata.setdefault("git_commit", current_git_commit())
    metadata = _json_safe(metadata)

    files_written: list[str] = []
    skipped_tables: list[dict[str, str]] = []
    row_counts: dict[str, int] = {}
    column_names: dict[str, list[str]] = {}

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(_json_dumps(metadata), encoding="utf-8")
    files_written.append(metadata_path.name)

    for table_name, table in tables.items():
        safe_name = safe_filename_stem(table_name)
        if table is None:
            skipped_tables.append({"table": safe_name, "reason": "missing"})
            continue
        if not isinstance(table, pd.DataFrame):
            skipped_tables.append({"table": safe_name, "reason": "not_dataframe"})
            continue
        if table.empty:
            skipped_tables.append({"table": safe_name, "reason": "empty"})
            continue

        frame = table.copy(deep=True)
        output_name = f"{safe_name}.csv"
        frame.to_csv(run_dir / output_name, index=False)
        files_written.append(output_name)
        row_counts[output_name] = int(len(frame))
        column_names[output_name] = [str(column) for column in frame.columns]

    manifest = {
        "run_folder": str(run_dir),
        "created_at": created_at_text,
        "files_written": sorted(files_written),
        "tables_skipped": skipped_tables,
        "row_counts": row_counts,
        "column_names": column_names,
        "app_settings": _json_safe(run_metadata.get("app_settings", {})),
        "parameters": _json_safe(_metadata_parameters(run_metadata)),
    }

    handoff = build_codex_handoff(
        run_metadata=metadata,
        tables=tables,
        manifest=manifest,
        notes=notes or {},
    )
    handoff_path = run_dir / "codex_handoff.md"
    handoff_path.write_text(handoff, encoding="utf-8")
    files_written.append(handoff_path.name)

    manifest["files_written"] = sorted(files_written + ["diagnostics_manifest.json"])
    manifest_path = run_dir / "diagnostics_manifest.json"
    manifest_path.write_text(_json_dumps(manifest), encoding="utf-8")

    latest_dir = root / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    return ResearchExportResult(
        run_dir=run_dir,
        latest_dir=latest_dir,
        manifest=manifest,
        codex_handoff=handoff,
    )


def safe_filename_stem(value: object) -> str:
    """Return a stable snake_case filename stem."""

    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "table"
    if not SAFE_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Unsafe export filename stem: {value!r}")
    return name


def zip_research_bundle(run_dir: str | Path) -> bytes:
    """Return a zip archive of a Research Lab bundle without writing another file."""

    root = Path(run_dir)
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(root))
    return buffer.getvalue()


def build_codex_handoff(
    *,
    run_metadata: Mapping[str, object],
    tables: Mapping[str, object],
    manifest: Mapping[str, object],
    notes: Mapping[str, str] | None = None,
) -> str:
    """Build the Markdown handoff read by the next Codex iteration."""

    notes = notes or {}
    target_quality = _frame(tables.get("target_quality_summary"))
    feature_group = _first_non_empty(
        tables.get("target_feature_group_comparison"),
        tables.get("feature_group_comparison"),
    )
    regime = _first_non_empty(
        tables.get("target_regime_comparison"),
        tables.get("regime_segmented_ml_diagnostics"),
    )
    diagnostics_summary = _frame(tables.get("ml_diagnostics_summary"))
    ml_reliability = _frame(tables.get("ml_reliability_by_regime"))
    ml_reliability_gate = _frame(tables.get("ml_reliability_gate_diagnostics"))
    momentum_quality = _frame(tables.get("momentum_quality_diagnostics"))
    validation_leakage = _frame(tables.get("validation_leakage_diagnostics"))
    validation_fold_stability = _frame(tables.get("validation_fold_stability"))
    validation_overfit = _frame(tables.get("validation_overfit_warnings"))
    portfolio_crowding = _frame(tables.get("portfolio_crowding_summary"))
    factor_crowding = _frame(tables.get("portfolio_factor_crowding_summary"))
    earnings_pead = _frame(tables.get("earnings_pead_summary"))
    drawdown_quality = _frame(tables.get("drawdown_risk_calibration_quality"))
    feature_audit = _frame(tables.get("feature_audit_summary"))
    redundancy = _frame(tables.get("feature_redundancy_selection"))
    importance_stability = _frame(tables.get("feature_importance_stability"))

    lines = [
        "# Stock Signal Lab Research Run Handoff",
        "",
        "## Run metadata",
        f"- Created at: {_display(run_metadata.get('created_at'))}",
        f"- Benchmark: {_display(run_metadata.get('benchmark'))}",
        f"- Feature group: {_display(run_metadata.get('feature_group'))}",
        f"- Model mode: {_display(run_metadata.get('model_mode'))}",
        (
            "- Train/test/step/embargo: "
            f"{_display(run_metadata.get('train_window'))} / "
            f"{_display(run_metadata.get('test_window'))} / "
            f"{_display(run_metadata.get('step_size'))} / "
            f"{_display(run_metadata.get('embargo_effective', run_metadata.get('embargo_requested')))}"
        ),
        f"- Ticker count: {_display(run_metadata.get('ticker_count'))}",
        "",
        "## Main evidence",
        _main_evidence(diagnostics_summary, manifest),
        "",
        "## Production target status",
        _production_target_status(target_quality),
        "",
        "## Alternative target candidates",
        _alternative_target_candidates(target_quality),
        "",
        "## Regime warnings",
        _regime_warnings(regime, target_quality),
        "",
        "## ML reliability by regime",
        _ml_reliability_evidence(ml_reliability),
        "",
        "## ML reliability gate diagnostics",
        _ml_reliability_gate_evidence(ml_reliability_gate),
        "",
        "## Momentum quality diagnostics",
        _momentum_quality_evidence(momentum_quality),
        "",
        "## Validation leakage / overfit diagnostics",
        _validation_evidence(validation_leakage, validation_fold_stability, validation_overfit),
        "",
        "## Portfolio crowding diagnostics",
        _portfolio_crowding_evidence(portfolio_crowding, factor_crowding),
        "",
        "## Earnings / PEAD diagnostics",
        _earnings_pead_evidence(earnings_pead),
        "",
        "## Feature group findings",
        _feature_group_findings(feature_group, target_quality),
        "",
        "## Calibration warnings",
        _calibration_warnings(target_quality, drawdown_quality),
        "",
        "## Drawdown-risk evidence",
        _drawdown_evidence(drawdown_quality, _frame(tables.get("drawdown_risk_calibration"))),
        "",
        "## Feature diagnostics",
        _feature_diagnostics(feature_audit, redundancy, importance_stability),
        "",
        "## Suggested next engineering direction",
        _suggested_next_direction(target_quality, notes),
        "",
        "## How Codex should use this bundle",
        "- Read this handoff and the exported CSV files before coding.",
        "- Do not rely on screenshots as Research Lab evidence.",
        "- Use `python -m src.research.run_research_lab` for fresh headless evidence.",
        "- Compare before/after bundles before claiming an algorithmic improvement.",
        "- Do not claim improvement unless metrics and stop rules support it.",
        "- If target-quality evidence says keep baseline, no production target switch is supported.",
        "",
        "## Codex instructions for next iteration",
        "- Read this handoff and the CSV files in this folder.",
        "- Do not change production target unless explicitly instructed.",
        "- Do not change ML Score formula unless explicitly instructed.",
        "- Use the evidence here to propose the next coherent enhancement.",
        "",
    ]
    return "\n".join(lines)


def _unique_run_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def _json_dumps(payload: object) -> str:
    return json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _metadata_parameters(metadata: Mapping[str, object]) -> dict[str, object]:
    keys = [
        "feature_group",
        "model_mode",
        "model_name",
        "train_window",
        "test_window",
        "step_size",
        "embargo_requested",
        "embargo_effective",
        "classification_threshold",
        "target_candidates_enabled",
        "extended_target_comparison_enabled",
        "data_start",
        "data_end",
    ]
    return {key: metadata[key] for key in keys if key in metadata}


def _frame(value: object) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _first_non_empty(*values: object) -> pd.DataFrame:
    for value in values:
        frame = _frame(value)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _display(value: object) -> str:
    safe = _json_safe(value)
    if safe in (None, ""):
        return "Unavailable"
    if isinstance(safe, list):
        return ", ".join(str(item) for item in safe)
    return str(safe)


def _main_evidence(summary: pd.DataFrame, manifest: Mapping[str, object]) -> str:
    row_counts = manifest.get("row_counts", {})
    if summary.empty:
        count = len(row_counts) if isinstance(row_counts, Mapping) else 0
        return f"This run exported {count} non-empty diagnostics tables for review."
    snippets: list[str] = []
    for _, row in summary.head(4).iterrows():
        target = row.get("target", row.get("metric", "diagnostic"))
        metrics = []
        for column in ("roc_auc", "pr_auc", "accuracy", "brier_score", "bucket_spread"):
            if column in row and pd.notna(row[column]):
                metrics.append(f"{column}={row[column]}")
        snippets.append(f"{target}: {', '.join(metrics) if metrics else 'summary available'}")
    return "This run suggests the main exported diagnostics to inspect are: " + "; ".join(snippets) + "."


def _production_target_status(target_quality: pd.DataFrame) -> str:
    if target_quality.empty:
        return "No target_quality_summary was exported, so production target quality was not assessed in this bundle."
    baseline = _baseline_target_rows(target_quality)
    row = baseline.iloc[0] if not baseline.empty else target_quality.iloc[0]
    status = row.get("production_candidate_status", row.get("overall_target_quality", "available"))
    step = row.get("recommended_next_step", "review exported target quality evidence")
    target = row.get("target_id", "production baseline")
    return f"This run suggests {target} remains a research-reviewed target with status {status}. Recommended next step: {step}."


def _alternative_target_candidates(target_quality: pd.DataFrame) -> str:
    if target_quality.empty:
        return "No alternative target ranking was exported."
    alternatives = target_quality.drop(_baseline_target_rows(target_quality).index, errors="ignore")
    if alternatives.empty:
        return "No alternative target rows were exported."
    ranked = _rank_targets(alternatives)
    strongest = ranked.head(2)["target_id"].astype(str).tolist() if "target_id" in ranked else []
    weakest = ranked.tail(2)["target_id"].astype(str).tolist() if "target_id" in ranked else []
    return (
        "Candidate for further research, not production. "
        f"Strongest exported candidates: {', '.join(strongest) or 'Unavailable'}. "
        f"Weakest exported candidates: {', '.join(weakest) or 'Unavailable'}."
    )


def _regime_warnings(regime: pd.DataFrame, target_quality: pd.DataFrame) -> str:
    warnings: list[str] = []
    if "regime_stability" in target_quality:
        unstable = target_quality[
            target_quality["regime_stability"].astype(str).str.contains("weak|unstable|mixed|sensitive", case=False, na=False)
        ]
        warnings.extend(unstable.get("target_id", pd.Series(dtype=str)).astype(str).head(3).tolist())
    if "direction" in regime:
        inverted = regime[regime["direction"].astype(str).str.contains("invert|negative", case=False, na=False)]
        warnings.extend(inverted.get("target_id", inverted.get("regime", pd.Series(dtype=str))).astype(str).head(3).tolist())
    if warnings:
        return "Evidence is mixed across regimes for: " + ", ".join(dict.fromkeys(warnings)) + "."
    return "No explicit regime-sensitive or inverted behaviour was identified from the exported tables."


def _ml_reliability_evidence(reliability: pd.DataFrame) -> str:
    if reliability.empty:
        return "No ML reliability-by-regime table was exported."
    if "classification" not in reliability:
        return "This table shows where ML score historically worked, failed, or lacked enough evidence."

    counts = reliability["classification"].astype(str).value_counts().sort_index()
    count_text = ", ".join(f"{classification}={count}" for classification, count in counts.items())
    return (
        "This table shows where ML score historically worked, failed, or lacked enough evidence. "
        f"Exported {len(reliability)} regime rows"
        + (f": {count_text}." if count_text else ".")
    )


def _ml_reliability_gate_evidence(gate_diagnostics: pd.DataFrame) -> str:
    if gate_diagnostics.empty:
        return "No ML reliability gate diagnostics table was exported."
    if "classification" not in gate_diagnostics:
        return (
            "Research-only reliability gate diagnostics were exported. "
            "These do not change production scoring."
        )

    counts = gate_diagnostics["classification"].astype(str).value_counts().sort_index()
    count_text = ", ".join(f"{classification}={count}" for classification, count in counts.items())
    return (
        "Research-only reliability gates were tested without changing production scoring. "
        f"Exported {len(gate_diagnostics)} gate rows"
        + (f": {count_text}." if count_text else ".")
    )


def _momentum_quality_evidence(momentum_quality: pd.DataFrame) -> str:
    if momentum_quality.empty:
        return "No momentum quality diagnostics table was exported."
    if "classification" not in momentum_quality:
        return (
            "Research-only momentum quality diagnostics were exported. "
            "These do not change production scoring."
        )

    counts = momentum_quality["classification"].astype(str).value_counts().sort_index()
    count_text = ", ".join(f"{classification}={count}" for classification, count in counts.items())
    overall = (
        momentum_quality[momentum_quality["ticker"].astype(str) == "ALL"]
        if "ticker" in momentum_quality
        else pd.DataFrame()
    )
    bucket_spread = (
        _display(overall["bucket_spread"].dropna().iloc[0])
        if not overall.empty and "bucket_spread" in overall and overall["bucket_spread"].notna().any()
        else "Unavailable"
    )
    return (
        "Research-only momentum quality diagnostics checked steady, broad-based momentum versus fragile momentum. "
        f"Overall bucket spread: {bucket_spread}"
        + (f"; classifications: {count_text}." if count_text else ".")
    )


def _classification_counts(frame: pd.DataFrame) -> str:
    if frame.empty or "classification" not in frame:
        return ""
    counts = frame["classification"].astype(str).value_counts().sort_index()
    return ", ".join(f"{classification}={count}" for classification, count in counts.items())


def _validation_evidence(
    leakage: pd.DataFrame,
    fold_stability: pd.DataFrame,
    overfit: pd.DataFrame,
) -> str:
    if leakage.empty and fold_stability.empty and overfit.empty:
        return "No validation leakage or overfit diagnostics table was exported."

    parts = [
        "Research-only validation diagnostics checked fold gaps, fold instability, and thin or concentrated evidence without changing production scoring."
    ]
    leakage_counts = _classification_counts(leakage)
    if leakage_counts:
        parts.append(f"Leakage rows={len(leakage)}: {leakage_counts}.")
    stability_counts = _classification_counts(fold_stability)
    if stability_counts:
        parts.append(f"Fold-stability rows={len(fold_stability)}: {stability_counts}.")
    overfit_counts = _classification_counts(overfit)
    if overfit_counts:
        parts.append(f"Overfit-warning rows={len(overfit)}: {overfit_counts}.")
    return " ".join(parts)


def _earnings_pead_evidence(earnings_pead: pd.DataFrame) -> str:
    if earnings_pead.empty:
        return "No earnings / PEAD diagnostics table was exported."
    if "classification" not in earnings_pead:
        return (
            "Research-only earnings-window diagnostics were exported. "
            "These do not change production scoring."
        )

    counts = earnings_pead["classification"].astype(str).value_counts().sort_index()
    count_text = ", ".join(f"{classification}={count}" for classification, count in counts.items())
    direction = _display(earnings_pead.iloc[0].get("pead_signal_direction"))
    ml_effect = _display(earnings_pead.iloc[0].get("ml_near_earnings_effect"))
    return (
        "Research-only earnings / PEAD diagnostics were exported without changing production scoring. "
        f"PEAD direction: {direction}; ML near earnings effect: {ml_effect}"
        + (f"; classifications: {count_text}." if count_text else ".")
    )


def _portfolio_crowding_evidence(portfolio_crowding: pd.DataFrame, factor_crowding: pd.DataFrame) -> str:
    if portfolio_crowding.empty and factor_crowding.empty:
        return "No portfolio crowding diagnostics table was exported."
    parts = [
        "Research-only portfolio crowding diagnostics checked correlation overlap and factor proxy concentration without changing actions or sizing."
    ]
    if not portfolio_crowding.empty:
        row = portfolio_crowding.iloc[0]
        parts.append(
            "Correlation crowding: "
            f"{_display(row.get('classification'))}; "
            f"high-overlap pairs={_display(row.get('high_overlap_pair_count'))}; "
            f"largest cluster={_display(row.get('largest_cluster_size'))}."
        )
    if not factor_crowding.empty and "classification" in factor_crowding:
        counts = factor_crowding["classification"].astype(str).value_counts().sort_index()
        count_text = ", ".join(f"{classification}={count}" for classification, count in counts.items())
        parts.append(f"Factor proxy rows={len(factor_crowding)}" + (f": {count_text}." if count_text else "."))
    return " ".join(parts)


def _feature_group_findings(feature_group: pd.DataFrame, target_quality: pd.DataFrame) -> str:
    if "best_feature_group" in target_quality and not target_quality.empty:
        best = target_quality["best_feature_group"].dropna().astype(str)
        if not best.empty:
            return "This run suggests these feature groups deserve review: " + ", ".join(best.head(4).tolist()) + "."
    if "feature_group" in feature_group and not feature_group.empty:
        return "Feature-group comparison was exported for: " + ", ".join(feature_group["feature_group"].dropna().astype(str).unique()) + "."
    return "No feature-group comparison table was exported."


def _calibration_warnings(target_quality: pd.DataFrame, drawdown_quality: pd.DataFrame) -> str:
    warnings: list[str] = []
    if "calibration_quality" in target_quality:
        poor = target_quality[
            target_quality["calibration_quality"].astype(str).str.contains("poor|weak|bad", case=False, na=False)
        ]
        warnings.extend(poor.get("target_id", pd.Series(dtype=str)).astype(str).head(3).tolist())
    if "calibration_gap" in target_quality:
        high_gap = target_quality[pd.to_numeric(target_quality["calibration_gap"], errors="coerce") >= 0.15]
        warnings.extend(high_gap.get("target_id", pd.Series(dtype=str)).astype(str).head(3).tolist())
    if not drawdown_quality.empty:
        warnings.append("drawdown-risk calibration quality exported")
    if warnings:
        return "Review calibration before production changes: " + ", ".join(dict.fromkeys(warnings)) + "."
    return "No major calibration warning was identified from the exported tables."


def _drawdown_evidence(drawdown_quality: pd.DataFrame, drawdown_calibration: pd.DataFrame) -> str:
    if drawdown_quality.empty and drawdown_calibration.empty:
        return "No drawdown-risk calibration evidence was exported."
    if not drawdown_quality.empty:
        return f"Drawdown-risk calibration quality was exported with {len(drawdown_quality)} rows for review."
    return f"Drawdown-risk calibration buckets were exported with {len(drawdown_calibration)} rows for review."


def _feature_diagnostics(
    feature_audit: pd.DataFrame,
    redundancy: pd.DataFrame,
    importance_stability: pd.DataFrame,
) -> str:
    parts: list[str] = []
    if not feature_audit.empty:
        parts.append(f"feature audit summary rows={len(feature_audit)}")
    if not redundancy.empty:
        parts.append(f"redundancy selection rows={len(redundancy)}")
    if not importance_stability.empty:
        counts = importance_stability["classification"].value_counts().to_dict()
        count_text = ", ".join(f"{key}={value}" for key, value in counts.items())
        parts.append(f"importance stability rows={len(importance_stability)} ({count_text})")
    if parts:
        return "Exported feature diagnostics include " + ", ".join(parts) + "."
    return "No feature audit, redundancy, or importance-stability diagnostics were exported."


def _suggested_next_direction(target_quality: pd.DataFrame, notes: Mapping[str, str]) -> str:
    if "suggested_next_engineering_direction" in notes:
        return notes["suggested_next_engineering_direction"]
    if target_quality.empty:
        return "Do not switch production target yet. Review the exported diagnostics and rerun with target quality enabled if needed."
    status = target_quality.get(
        "production_candidate_status",
        pd.Series("", index=target_quality.index, dtype=str),
    )
    candidate = target_quality[
        status.astype(str).str.contains("trial|candidate|promising", case=False, na=False)
    ]
    if candidate.empty:
        return "Do not switch production target yet. Use this evidence to refine target definitions and investigate reliability state preview."
    targets = ", ".join(candidate.get("target_id", pd.Series(dtype=str)).astype(str).head(3).tolist())
    return f"This run suggests {targets} may deserve further research. Keep it as a production-trial candidate, not a production switch."


def _baseline_target_rows(target_quality: pd.DataFrame) -> pd.DataFrame:
    if target_quality.empty:
        return target_quality
    masks = []
    if "target_type" in target_quality:
        masks.append(target_quality["target_type"].astype(str).str.contains("production|baseline", case=False, na=False))
    if "target_id" in target_quality:
        masks.append(target_quality["target_id"].astype(str).str.contains("outperform_20d", case=False, na=False))
    if not masks:
        return target_quality.iloc[0:0]
    mask = masks[0]
    for item in masks[1:]:
        mask = mask | item
    return target_quality[mask]


def _rank_targets(target_quality: pd.DataFrame) -> pd.DataFrame:
    if "candidate_rank" in target_quality:
        return target_quality.sort_values("candidate_rank", kind="mergesort")
    quality_order = {
        "promising": 0,
        "usable": 1,
        "mixed": 2,
        "special setup only": 3,
        "unusable": 4,
    }
    ranked = target_quality.copy()
    ranked["_quality_rank"] = (
        ranked.get("overall_target_quality", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .map(quality_order)
        .fillna(99)
    )
    return ranked.sort_values("_quality_rank", kind="mergesort").drop(columns=["_quality_rank"])
