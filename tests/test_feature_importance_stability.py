from __future__ import annotations

import pandas as pd

from src.ml.diagnostics import (
    build_feature_family_importance_stability,
    build_feature_importance_stability,
)


def fold_importance(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {"target": "outperformance", "selected_model": "regularized_logistic"}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_feature_importance_stability_classifies_repeated_top_feature() -> None:
    data = fold_importance(
        [
            {"fold": fold, "feature": "return_20d", "abs_importance": value}
            for fold, value in enumerate([1.0, 1.1, 0.9, 1.0], start=1)
        ]
        + [
            {"fold": fold, "feature": "volatility_20d", "abs_importance": 0.1}
            for fold in range(1, 5)
        ]
    )

    result = build_feature_importance_stability(data)

    stable = result[result["feature"].eq("return_20d")].iloc[0]
    assert stable["classification"] == "stable"
    assert stable["fold_count"] == 4
    assert stable["top_decile_fold_count"] == 4


def test_feature_importance_stability_flags_one_fold_dominance() -> None:
    data = fold_importance(
        [
            {"fold": 1, "feature": "return_20d", "abs_importance": 2.0},
            {"fold": 2, "feature": "return_20d", "abs_importance": 0.0},
            {"fold": 3, "feature": "return_20d", "abs_importance": 0.0},
            {"fold": 4, "feature": "return_20d", "abs_importance": 0.0},
        ]
    )

    result = build_feature_importance_stability(data)

    assert result.loc[0, "classification"] == "unstable"
    assert result.loc[0, "top_decile_fold_count"] == 1


def test_feature_importance_stability_handles_insufficient_folds() -> None:
    data = fold_importance(
        [
            {"fold": 1, "feature": "return_20d", "abs_importance": 1.0},
            {"fold": 2, "feature": "return_20d", "abs_importance": 1.0},
        ]
    )

    result = build_feature_importance_stability(data)

    assert result.loc[0, "classification"] == "insufficient_data"


def test_feature_importance_stability_handles_unavailable_importance() -> None:
    result = build_feature_importance_stability(pd.DataFrame())

    assert result.loc[0, "classification"] == "unavailable"
    assert result.loc[0, "feature"] == "unavailable"


def test_feature_importance_stability_adds_selected_model_aggregate() -> None:
    data = pd.DataFrame(
        [
            {
                "target": "outperformance",
                "selected_model": "regularized_logistic" if fold % 2 else "random_forest_shallow",
                "fold": fold,
                "feature": "return_20d",
                "abs_importance": 1.0,
            }
            for fold in range(1, 5)
        ]
        + [
            {
                "target": "outperformance",
                "selected_model": "regularized_logistic" if fold % 2 else "random_forest_shallow",
                "fold": fold,
                "feature": "volatility_20d",
                "abs_importance": 0.1,
            }
            for fold in range(1, 5)
        ]
    )

    result = build_feature_importance_stability(data)

    aggregate = result[result["model"].eq("all_selected_models") & result["feature"].eq("return_20d")].iloc[0]
    assert aggregate["classification"] == "stable"
    assert aggregate["fold_count"] == 4


def test_feature_family_importance_stability_aggregates_families() -> None:
    data = fold_importance(
        [
            {"fold": fold, "feature": "return_20d", "abs_importance": 0.7}
            for fold in range(1, 5)
        ]
        + [
            {"fold": fold, "feature": "momentum_60d", "abs_importance": 0.4}
            for fold in range(1, 5)
        ]
        + [
            {"fold": fold, "feature": "volatility_20d", "abs_importance": 0.1}
            for fold in range(1, 5)
        ]
    )

    result = build_feature_family_importance_stability(data)

    family = result[result["feature_family"].eq("momentum / return")].iloc[0]
    assert family["classification"] == "stable"
    assert family["feature_count"] == 2
    assert family["top_family_fold_count"] == 4
