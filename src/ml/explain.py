"""Simple model interpretation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def feature_importance_frame(model: Pipeline, feature_columns: list[str], top_n: int = 20) -> pd.DataFrame:
    """Return coefficients or impurity importances when the fitted model exposes them."""

    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
        kind = "scaled_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.ravel(estimator.feature_importances_)
        kind = "feature_importance"
    else:
        return pd.DataFrame(columns=["feature", "importance", "abs_importance", "kind"])

    output = pd.DataFrame({"feature": feature_columns, "importance": values})
    output["abs_importance"] = output["importance"].abs()
    output["kind"] = kind
    return output.sort_values("abs_importance", ascending=False).head(top_n).reset_index(drop=True)


def feature_group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importance by technical/Fourier/wavelet group."""

    if importance.empty:
        return pd.DataFrame()

    def group_name(feature: str) -> str:
        if feature.startswith("fourier_"):
            return "Fourier"
        if feature.startswith("wavelet_"):
            return "Wavelet"
        return "Technical"

    grouped = importance.copy()
    grouped["feature_group"] = grouped["feature"].map(group_name)
    return (
        grouped.groupby("feature_group", observed=True)["abs_importance"]
        .sum()
        .reset_index(name="total_abs_importance")
        .sort_values("total_abs_importance", ascending=False)
    )

