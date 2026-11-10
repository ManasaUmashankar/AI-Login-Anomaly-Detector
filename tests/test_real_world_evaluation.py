import pandas as pd
import pytest

from src.preprocessing import REQUIRED_COLUMNS
from src.real_world_evaluation import evaluate_real_world_csv


def test_external_evaluation_uses_chronological_holdout(tmp_path):
    rows = []
    for i in range(40):
        rows.append([
            (pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            f"user-{i % 4}", f"203.0.113.{10 + i % 4}", "Bengaluru, India",
            "Windows Laptop", "failed" if i % 11 == 0 else "success",
            5 if i % 11 == 0 else 0, "test-agent", f"session-{i}",
            "anomaly" if i % 11 == 0 else "normal",
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


def test_external_evaluation_requires_independent_label_provenance(tmp_path):
    frame = pd.DataFrame(columns=REQUIRED_COLUMNS + ["true_label"])
    path = tmp_path / "empty.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label source"):
        evaluate_real_world_csv(path, label_provenance="synthetic labels")
