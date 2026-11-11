import pandas as pd
import pytest

from src.preprocessing import REQUIRED_COLUMNS
from src.real_world_evaluation import evaluate_real_world_csv


def test_external_evaluation_uses_chronological_holdout(tmp_path):
    rows = []
    for i in range(40):
        is_anomaly = i % 3 == 0
        rows.append([
            (pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            f"user-{i % 4}", f"203.0.113.{10 + i % 4}", "Bengaluru, India",
            "Windows Laptop", "failed" if is_anomaly else "success",
            5 if is_anomaly else 0, "test-agent", f"session-{i}",
            "anomaly" if is_anomaly else "normal",
        ])
    path = tmp_path / "external_auth.csv"
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS + ["true_label"]).to_csv(path, index=False)

    report = evaluate_real_world_csv(
        path,
        label_provenance="Manually reviewed incident tickets and analyst labels",
        train_fraction=0.60,
    )

    assert report["evaluation_type"] == "independently_labelled_external_authentication_logs"
    assert report["training_rows"] + report["holdout_rows"] == 40
    assert report["training_end"] < report["holdout_start"]
    assert set(report["metrics"]) == {"isolation_forest", "rules", "hybrid"}
    assert report["holdout_label_counts"]["normal"] >= 2
    assert report["holdout_label_counts"]["anomaly"] >= 2
    assert len(report["dataset_sha256"]) == 64


def test_external_evaluation_requires_independent_label_provenance(tmp_path):
    frame = pd.DataFrame(columns=REQUIRED_COLUMNS + ["true_label"])
    path = tmp_path / "empty.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label source"):
        evaluate_real_world_csv(path, label_provenance="Labels produced by the same synthetic generator")


def test_external_evaluation_rejects_holdout_without_both_classes(tmp_path):
    rows = []
    for i in range(30):
        rows.append([
            (pd.Timestamp("2026-02-01") + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            f"user-{i % 3}", f"203.0.113.{10 + i % 3}", "Bengaluru, India",
            "Windows Laptop", "success", 0, "test-agent", f"normal-{i}", "normal",
        ])
    path = tmp_path / "normal_only.csv"
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS + ["true_label"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="holdout must contain"):
        evaluate_real_world_csv(
            path,
            label_provenance="Manually reviewed authentication events",
            train_fraction=0.60,
        )
