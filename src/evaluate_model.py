import math

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
    cleaned = preprocess_logs(working)
    if len(cleaned) < 4:
        raise ValueError("At least four valid login rows are required for a temporal holdout.")

    features = build_features(cleaned)
    split_at = math.floor(len(features) * train_fraction)
    split_at = min(max(split_at, 2), len(features) - 1)

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


def main():
    df = load_logs(include_labels=True)
    metrics, split = evaluate_temporal_holdout(df)

    print("\n========== CHRONOLOGICAL HOLDOUT EVALUATION ==========")
    print("The model is trained on earlier events and evaluated on later unseen events.")
    print(f"Total valid logins:      {split['total_rows']}")
    print(f"Training logins:         {split['train_rows']}")
    print(f"Holdout logins:          {split['test_rows']}")
    print(f"Training ends:           {split['train_end']}")
    print(f"Holdout starts:          {split['test_start']}")
    print(f"Actual holdout anomalies: {(split['y_true'] == 'anomaly').sum()}")
    print(f"Predicted anomalies:      {(split['y_pred'] == 'anomaly').sum()}")

    print("\n========== PERFORMANCE METRICS ==========")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1-score:            {metrics['f1']:.4f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.4f}")

    print("\n========== CONFUSION MATRIX ==========")
    print("Rows = actual [normal, anomaly]; columns = predicted [normal, anomaly]")
    print(metrics["confusion_matrix"])

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(
        split["y_true"],
        split["y_pred"],
        labels=["normal", "anomaly"],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    ))
    print("Evaluation completed. Results describe this synthetic temporal holdout, not real-world performance.")


if __name__ == "__main__":
    main()
