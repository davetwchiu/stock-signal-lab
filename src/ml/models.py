"""Small scikit-learn model factories for Stage 2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_OPTIONS = ("logistic_regression", "random_forest", "hist_gradient_boosting")


def build_model_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Build a leakage-safe sklearn Pipeline."""

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", model)])
    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=random_state)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", model)])
    raise ValueError(f"Unknown model_name: {model_name}")


def fit_classifier(
    data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    random_state: int = 42,
) -> Pipeline:
    """Fit a classifier, using a dummy fallback when only one class is available."""

    training = data.dropna(subset=[label_column])
    if training.empty:
        training = pd.DataFrame([{column: 0.0 for column in feature_columns}])
        y = pd.Series([0])
    else:
        y = training[label_column].astype(int)
    if y.nunique() < 2:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", DummyClassifier(strategy="constant", constant=int(y.iloc[0]) if len(y) else 0)),
            ]
        )
    else:
        pipeline = build_model_pipeline(model_name, random_state=random_state)
    pipeline.fit(training[feature_columns], y)
    return pipeline


def predict_positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for fitted sklearn classifiers."""

    probabilities = model.predict_proba(features)
    classes = list(model.named_steps["model"].classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(len(features), dtype=float)
