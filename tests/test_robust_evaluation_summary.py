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


def test_supervised_category_prior_uses_training_labels_only():
    from src.robust_evaluation import _supervised_category_prior

    train = pd.DataFrame({
        "user_agent": ["known-bad", "known-bad", "ordinary", "ordinary"],
        "true_label": ["anomaly", "anomaly", "normal", "normal"],
    })
    test = pd.DataFrame({"user_agent": ["known-bad", "ordinary", "unseen"]})

    predictions = _supervised_category_prior(train, test, "user_agent")

    assert predictions.tolist() == [1, 0, 0]
    # Changing or adding holdout labels cannot affect predictions: they are not an input.
    test_with_labels = test.assign(true_label=["normal", "anomaly", "anomaly"])
    assert _supervised_category_prior(train, test_with_labels, "user_agent").tolist() == [1, 0, 0]


def test_supervised_category_prior_rejects_unknown_training_labels():
    from src.robust_evaluation import _supervised_category_prior

    train = pd.DataFrame({"device": ["x"], "true_label": ["unknown"]})
    test = pd.DataFrame({"device": ["x"]})

    with pytest.raises(ValueError, match="normal/anomaly"):
        _supervised_category_prior(train, test, "device")


def test_single_seed_summary_keeps_sample_std_undefined():
    from src.robust_evaluation import summarize_seed_metrics

    frame = pd.DataFrame([{
        "seed": 7, "method": "ml_only", "precision": 0.8,
        "recall": 0.7, "f1": 0.74, "pr_auc": 0.75,
    }])
    summary = summarize_seed_metrics(frame)

    assert pd.isna(summary.loc["ml_only", "precision_std"])
    assert pd.isna(summary.loc["ml_only", "recall_std"])
    assert pd.isna(summary.loc["ml_only", "f1_std"])
    assert pd.isna(summary.loc["ml_only", "pr_auc_std"])


def test_summary_csv_uses_explicit_na_token_for_undefined_values(tmp_path):
    from src.robust_evaluation import summarize_seed_metrics

    frame = pd.DataFrame([
        {"seed": 1, "method": "hybrid", "precision": 0.8, "recall": 0.7, "f1": 0.74, "pr_auc": float("nan")},
        {"seed": 2, "method": "hybrid", "precision": 0.9, "recall": 0.8, "f1": 0.84, "pr_auc": float("nan")},
    ])
    path = tmp_path / "summary.csv"
    summarize_seed_metrics(frame).to_csv(path, na_rep="NA")
    loaded = pd.read_csv(path, na_values=["NA"])

    assert loaded.columns.is_unique
    assert pd.isna(loaded.loc[0, "pr_auc_mean"])
