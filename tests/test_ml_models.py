from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.models import available_model_candidates, build_classifier, predict_positive_probability


class ReversedClassModel:
    classes_ = np.array([1, 0])


class ReversedClassPipeline:
    named_steps = {"model": ReversedClassModel()}

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                [0.25, 0.75],
                [0.80, 0.20],
            ]
        )[: len(features)]


def test_predict_positive_probability_uses_class_one_probability() -> None:
    features = pd.DataFrame({"feature": [1.0, 2.0]})

    probabilities = predict_positive_probability(ReversedClassPipeline(), features)

    assert probabilities.tolist() == [0.25, 0.80]


def test_model_candidate_registry_includes_current_default() -> None:
    candidates = available_model_candidates()

    assert candidates[0] == "current_default"
    assert "current_default" in candidates
    assert build_classifier("current_default").named_steps["model"].__class__.__name__ == "LogisticRegression"
