import pandas as pd
import pytest

from src.evaluation_suite import build_overview_rows, run_evaluation_suite

METRICS = ("precision", "recall", "f1", "false_positive_rate", "accuracy")


def fake_metrics(value):
    return {**{name: value for name in METRICS}, "tp": 2, "fp": 1, "fn": 1, "tn": 6}


def test_build_overview_rows_compares_temporal_event_and_scenario_scopes():
    seed_result = {"summary": {
        "metrics": {name: {"mean": 0.8, "std": 0.1} for name in METRICS},
        "baseline_metrics": {name: {"mean": 0.9, "std": 0.0} for name in METRICS},
    }}
    stress = {"ml": fake_metrics(0.7), "rules": fake_metrics(0.8), "hybrid": fake_metrics(0.75)}
    scenarios = {"ml": fake_metrics(0.6), "rule": fake_metrics(0.7), "hybrid": fake_metrics(0.65)}

    frame = pd.DataFrame(build_overview_rows(seed_result, stress, scenarios))
    assert set(frame["evaluation_scope"]) == {
        "temporal_multi_seed_window_mean",
        "controlled_stress_event_level",
        "controlled_stress_scenario_level",
    }
    assert len(frame) == 25
    selected = frame[
        (frame["evaluation_scope"] == "controlled_stress_event_level")
        & (frame["detector"] == "Hybrid (ML OR rules)")
        & (frame["metric"] == "f1")
    ]
    assert selected.iloc[0]["mean"] == pytest.approx(0.75)


def test_run_evaluation_suite_requires_ground_truth_labels(tmp_path):
    with pytest.raises(ValueError, match="true_label"):
        run_evaluation_suite(pd.DataFrame({"timestamp": []}), report_dir=tmp_path)
