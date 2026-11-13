import pandas as pd

from src.model_comparison import evaluate_model_comparison
from src.preprocessing import REQUIRED_COLUMNS


def test_model_comparison_uses_three_models_and_chronological_windows():
    rows = []
    for index in range(48):
        rows.append([
            (pd.Timestamp("2026-01-01") + pd.Timedelta(hours=index)).strftime("%Y-%m-%d %H:%M:%S"),
            f"user-{index % 4}",
            f"203.0.113.{10 + (index % 4)}",
            "Bengaluru, India",
            "Windows Laptop",
            "failed" if index % 13 == 0 else "success",
            5 if index % 13 == 0 else 0,
            "test-agent",
            f"session-{index}",
            "anomaly" if index % 13 == 0 else "normal",
        ])
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS + ["true_label"])

    result = evaluate_model_comparison(frame, train_fraction=0.40, windows=2, seed=7)

    assert set(result["model"]) == {
        "Isolation Forest", "Local Outlier Factor", "One-Class SVM"
    }
    assert set(result["window"]) == {1, 2}
    assert (result["train_rows"] < result["test_rows"] + result["train_rows"]).all()
    assert result[["precision", "recall", "f1", "accuracy"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()
    assert (result["train_end"] < result["test_start"]).all()
