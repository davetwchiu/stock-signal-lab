from __future__ import annotations

import json

from src.experiments.logger import append_experiment_index, write_experiment_log
from src.experiments.schema import ExperimentRecord


def test_experiment_log_writing(tmp_path) -> None:
    record = ExperimentRecord(
        ticker_universe=["AAA", "BBB"],
        date_range=("2024-01-01", "2024-12-31"),
        feature_group="all",
        model_type="logistic_regression",
        label_type="label_outperform_20d",
        threshold_settings={"score": 0.6},
        transaction_cost_assumptions={"cost_bps": 5},
        validation_settings={"train_window": 60},
        portfolio_settings={"cash_floor": 0.1},
        key_metrics={"Sharpe": 1.0},
        output_file_paths={},
    )

    path = write_experiment_log(record, tmp_path)
    index = append_experiment_index(path)

    assert path.exists()
    assert index.exists()
    assert json.loads(path.read_text())["feature_group"] == "all"

