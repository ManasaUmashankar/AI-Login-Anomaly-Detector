import pandas as pd
import pytest

from src.audit_label_alignment import audit_label_alignment, binary_metrics


def test_binary_metrics_counts_and_scores():
    result = binary_metrics(
        ["normal", "normal", "anomaly", "anomaly"],
        ["normal", "anomaly", "anomaly", "normal"],
    )
    assert (result["tn"], result["fp"], result["tp"], result["fn"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["accuracy"] == 0.5


def test_binary_metrics_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        binary_metrics([], [])


def test_binary_metrics_rejects_unknown_labels():
    with pytest.raises(ValueError, match="only"):
        binary_metrics(["maybe"], ["normal"])


def test_audit_requires_ground_truth_labels():
    with pytest.raises(ValueError, match="true_label"):
        audit_label_alignment(pd.DataFrame({"timestamp": []}))


def test_audit_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        audit_label_alignment(pd.DataFrame(), risk_threshold=-1)
