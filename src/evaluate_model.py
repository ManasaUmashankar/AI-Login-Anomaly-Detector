import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.anomaly_detector import MODEL_FEATURES, apply_model, train_model
from src.detection_rules import apply_risk_scores, apply_rules
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs


def normalize_labels(labels):
    """Normalize synthetic ground-truth labels to normal/anomaly."""
    normalized = (
        pd.Series(labels)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"normal": "normal", "anomaly": "anomaly"})
    )
    if normalized.isna().any():
        values = pd.Series(labels).astype(str).unique().tolist()
        raise ValueError(f"Unexpected true_label values: {values}")
    return normalized.reset_index(drop=True)


def calculate_metrics(y_true, y_pred):
    """Calculate binary metrics with anomaly as the positive class."""
    y_true = normalize_labels(y_true)
    y_pred = normalize_labels(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("Actual labels and predictions must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty holdout set.")

    matrix = confusion_matrix(y_true, y_pred, labels=["normal", "anomaly"])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="anomaly", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="anomaly", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="anomaly", zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "confusion_matrix": matrix,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _temporal_group_starts(features):
    """Return row indices where a new timestamp group begins."""
    timestamps = features["timestamp"].reset_index(drop=True)
    if timestamps.empty:
        return []
    return [0] + [
        index for index in range(1, len(timestamps))
        if timestamps.iloc[index] != timestamps.iloc[index - 1]
    ]


def _choose_temporal_train_group(features, train_fraction, minimum_future_groups):
    """Choose a row-count-near split without separating equal timestamps."""
    group_starts = _temporal_group_starts(features)
    candidates = [
        group_index for group_index in range(1, len(group_starts))
        if group_starts[group_index] >= 2
        and len(group_starts) - group_index >= minimum_future_groups
    ]
    if not candidates:
        raise ValueError("Not enough distinct timestamps for the requested temporal split.")
    target_rows = len(features) * train_fraction
    return min(candidates, key=lambda group_index: abs(group_starts[group_index] - target_rows))


def evaluate_temporal_holdout(df, train_fraction=0.70):
    """Train on earlier logins and evaluate only on later, unseen logins."""
    if "true_label" not in df.columns:
        raise ValueError("Dataset must contain a true_label column for evaluation.")
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be at least 0.5 and less than 1.0.")

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    working["true_label"] = normalize_labels(working["true_label"]).values

    # Keep the labels only for scoring; preprocessing preserves the optional label.
    cleaned = preprocess_logs(working, verbose=False)
    if len(cleaned) < 4:
        raise ValueError("At least four valid login rows are required for a temporal holdout.")

    features = build_features(cleaned)
    group_starts = _temporal_group_starts(features)
    train_group = _choose_temporal_train_group(
        features, train_fraction, minimum_future_groups=1
    )
    split_at = group_starts[train_group]

    train_features = features.iloc[:split_at]
    test_features = features.iloc[split_at:]

    # Train only on the earlier period; never fit the model on holdout rows.
    model = train_model(train_features)
    predictions = apply_model(model, test_features)

    y_true = normalize_labels(test_features["true_label"])
    y_pred = predictions["ml_result"].astype(str).str.lower().map(
        {"normal": "normal", "anomalous": "anomaly"}
    )
    if y_pred.isna().any():
        raise ValueError(f"Unexpected prediction values: {predictions['ml_result'].unique().tolist()}")

    metrics = calculate_metrics(y_true, y_pred)
    metadata = {
        "total_rows": len(features),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "train_end": train_features["timestamp"].max(),
        "test_start": test_features["timestamp"].min(),
        "y_true": y_true,
        "y_pred": y_pred.reset_index(drop=True),
    }
    return metrics, metadata



def evaluate_temporal_windows(df, train_fraction=0.40, windows=4, random_state=None):
    """Evaluate consecutive future windows with an expanding training history."""
    if not isinstance(windows, int) or windows < 2:
        raise ValueError("windows must be an integer of at least 2.")
    if not 0.30 <= train_fraction < 0.90:
        raise ValueError("train_fraction must be at least 0.30 and less than 0.90.")
    if "true_label" not in df.columns:
        raise ValueError("Dataset must contain a true_label column for evaluation.")

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    working["true_label"] = normalize_labels(working["true_label"]).values
    cleaned = preprocess_logs(working)
    features = build_features(cleaned)

    group_starts = _temporal_group_starts(features)
    initial_train_group = _choose_temporal_train_group(
        features, train_fraction, minimum_future_groups=windows
    )
    remaining_groups = len(group_starts) - initial_train_group
    boundaries = [
        initial_train_group + (index * remaining_groups) // windows
        for index in range(windows + 1)
    ]

    results = []
    for index in range(windows):
        test_start = group_starts[boundaries[index]]
        test_end = (
            group_starts[boundaries[index + 1]]
            if boundaries[index + 1] < len(group_starts)
            else len(features)
        )

        train_features = features.iloc[:test_start]
        test_features = features.iloc[test_start:test_end]
        # Preserve existing monkeypatch/test behavior when no seed is supplied.
        model = (
            train_model(train_features)
            if random_state is None
            else train_model(train_features, random_state=random_state)
        )
        predictions = apply_model(model, test_features)

        y_true = normalize_labels(test_features["true_label"])
        y_pred = predictions["ml_result"].astype(str).str.lower().map(
            {"normal": "normal", "anomalous": "anomaly"}
        )
        if y_pred.isna().any():
            raise ValueError(
                f"Unexpected prediction values: {predictions['ml_result'].unique().tolist()}"
            )

        metrics = calculate_metrics(y_true, y_pred)

        # Compare the ML model with an explainable rule/risk-score baseline.
        rule_results = apply_risk_scores(apply_rules(test_features.copy()))
        baseline_pred = rule_results["risk_score"].ge(3).map(
            {True: "anomaly", False: "normal"}
        )
        baseline_metrics = calculate_metrics(y_true, baseline_pred)

        results.append({
            "window": index + 1,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "metadata": {
                "train_rows": len(train_features),
                "test_rows": len(test_features),
                "train_end": train_features["timestamp"].max(),
                "test_start": test_features["timestamp"].min(),
                "test_end": test_features["timestamp"].max(),
                "actual_anomalies": int((y_true == "anomaly").sum()),
                "predicted_anomalies": int((y_pred == "anomaly").sum()),
            },
        })

    return results



def evaluate_seed_sweep(df, seeds=range(20), train_fraction=0.40, windows=4):
    """Evaluate multiple model seeds and summarize mean/std of window-averaged metrics.

    Each seed uses the same chronological splits and labels; only the model's
    random_state changes. Results remain synthetic-data evaluation, not a
    real-world performance guarantee.
    """
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must contain at least one integer seed.")
    if any(not isinstance(seed, int) for seed in seed_list):
        raise ValueError("every seed must be an integer.")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique.")

    runs = {}
    for seed in seed_list:
        runs[seed] = evaluate_temporal_windows(
            df,
            train_fraction=train_fraction,
            windows=windows,
            random_state=seed,
        )

    metric_names = ("precision", "recall", "f1", "false_positive_rate", "accuracy")
    summary = {}
    for model_key in ("metrics", "baseline_metrics"):
        summary[model_key] = {}
        for metric in metric_names:
            per_seed_means = [
                sum(window[model_key][metric] for window in runs[seed]) / len(runs[seed])
                for seed in seed_list
            ]
            summary[model_key][metric] = {
                "mean": float(pd.Series(per_seed_means).mean()),
                "std": float(pd.Series(per_seed_means).std(ddof=1)) if len(per_seed_means) > 1 else 0.0,
                "per_seed_mean": dict(zip(seed_list, per_seed_means)),
            }

    return {"seeds": seed_list, "runs": runs, "summary": summary}


def print_seed_sweep_report(result):
    """Print the seed-sweep summary with mean and sample standard deviation."""
    print("\n========== MULTI-SEED TEMPORAL EVALUATION ==========")
    print(f"Seeds: {result['seeds']}")
    print("Values are mean +/- sample standard deviation across seed-level window means.")
    print("All results use synthetic data; this is not a real-world benchmark.")
    for metric in ("precision", "recall", "f1", "false_positive_rate", "accuracy"):
        ml = result["summary"]["metrics"][metric]
        baseline = result["summary"]["baseline_metrics"][metric]
        print(
            f"{metric.replace('_', ' ').title():22s} "
            f"ML {ml['mean']:.4f} +/- {ml['std']:.4f} | "
            f"Rule baseline {baseline['mean']:.4f} +/- {baseline['std']:.4f}"
        )

def print_temporal_window_report(results):
    """Print per-window metrics and an unweighted mean across windows."""
    metric_names = ("precision", "recall", "f1", "false_positive_rate", "accuracy")
    print("\n========== MULTI-WINDOW TEMPORAL EVALUATION ==========")
    print("Each window is evaluated using a model trained only on earlier rows.")
    print("All results are from synthetic data and are not a real-world benchmark.")

    for result in results:
        metrics = result["metrics"]
        meta = result["metadata"]
        print(f"\nWindow {result['window']}")
        print(f"  Training rows: {meta['train_rows']} (through {meta['train_end']})")
        print(f"  Holdout rows:  {meta['test_rows']} ({meta['test_start']} to {meta['test_end']})")
        print(f"  Actual anomalies: {meta['actual_anomalies']}")
        print(f"  Predicted anomalies: {meta['predicted_anomalies']}")
        print(
            "  Precision: {precision:.4f} | Recall: {recall:.4f} | "
            "F1: {f1:.4f} | FPR: {false_positive_rate:.4f} | "
            "Accuracy: {accuracy:.4f}".format(**metrics)
        )
        print("  Confusion matrix [actual normal, anomaly] x [predicted normal, anomaly]:")
        print(metrics["confusion_matrix"])
        baseline = result["baseline_metrics"]
        print(
            "  Rule baseline (risk score >= 3): "
            f"Precision {baseline['precision']:.4f} | Recall {baseline['recall']:.4f} | "
            f"F1 {baseline['f1']:.4f} | FPR {baseline['false_positive_rate']:.4f}"
        )

    print("\n========== UNWEIGHTED MEAN ACROSS WINDOWS ==========")
    for name in metric_names:
        mean_value = sum(item["metrics"][name] for item in results) / len(results)
        baseline_mean = sum(item["baseline_metrics"][name] for item in results) / len(results)
        print(
            f"{name.replace('_', ' ').title():22s} "
            f"ML {mean_value:.4f} | Rule baseline {baseline_mean:.4f}"
        )


def main():
    df = load_logs(include_labels=True)
    seed_results = evaluate_seed_sweep(df, seeds=range(20), train_fraction=0.40, windows=4)
    print_seed_sweep_report(seed_results)
    detail_seed = 42 if 42 in seed_results["runs"] else seed_results["seeds"][0]
    print(f"\nDetailed window metrics for seed {detail_seed}:")
    print_temporal_window_report(seed_results["runs"][detail_seed])


if __name__ == "__main__":
    main()
