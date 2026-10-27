import pandas as pd
import pytest

from src.robust_evaluation import summarize_seed_metrics


def test_summary_uses_unique_flat_metric_columns():
    per_seed = pd.DataFrame([
        {"seed": 0, "method": "hybrid", "precision": 0.8, "recall": 0.7, "f1": 0.74, "pr_auc": float("nan")},
        {"seed": 1, "method": "hybrid", "precision": 0.9, "recall": 0.8, "f1": 0.84, "pr_auc": float("nan")},
        {"seed": 0, "method": "ml_only", "precision": 0.7, "recall": 0.6, "f1": 0.64, "pr_auc": 0.75},
        {"seed": 1, "method": "ml_only", "precision": 0.8, "recall": 0.7, "f1": 0.74, "pr_auc": 0.85},
    ])

    summary = summarize_seed_metrics(per_seed)

    assert summary.columns.is_unique
    assert list(summary.columns) == [
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "pr_auc_mean", "pr_auc_std",
    ]
    assert summary.loc["hybrid", "precision_mean"] == pytest.approx(0.85)
    # PR-AUC is not computed for the hybrid method, so it stays undefined.
    assert pd.isna(summary.loc["hybrid", "pr_auc_mean"])
    assert pd.isna(summary.loc["hybrid", "pr_auc_std"])


def test_summary_rejects_missing_metric_columns():
    frame = pd.DataFrame([{"method": "hybrid", "precision": 0.8}])

    with pytest.raises(ValueError, match="Missing columns required"):
        summarize_seed_metrics(frame)
