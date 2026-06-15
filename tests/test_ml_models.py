from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.models import predict_positive_probability


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
