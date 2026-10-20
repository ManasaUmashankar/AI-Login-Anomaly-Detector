import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.read_dataset import load_logs
from src.anomaly_detector import run_pipeline


def main():
    # 1. Load the original dataset, including ground-truth labels.
    df = load_logs(include_labels=True)

    if "true_label" not in df.columns:
        raise ValueError(
            "The dataset has no 'true_label' column. "
            "Regenerate data/login_logs.csv using generate_dataset.py."
        )

    # 2. Normalize timestamps and labels.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    y_true = (
        df["true_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"normal": "normal", "anomaly": "anomaly"})
    )

    if y_true.isna().any():
        raise ValueError(
            "Unexpected values found in true_label. "
            f"Labels found: {df['true_label'].unique().tolist()}"
        )

    # 3. Run the complete detection pipeline.
    print("\nRunning login anomaly detection...\n")
    evaluation = run_pipeline()

    if "ml_result" not in evaluation.columns:
        raise ValueError(
            "Pipeline output is missing the 'ml_result' column."
        )

    # 4. Check that actual labels and predictions align.
    if len(y_true) != len(evaluation):
        raise ValueError(
            f"Row count mismatch: {len(y_true)} actual labels, "
            f"but {len(evaluation)} predictions. "
            "Check preprocessing and row ordering."
        )

    # 5. Convert model predictions to the same label format.
    y_pred = (
        evaluation["ml_result"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"normal": "normal", "anomalous": "anomaly"})
    )

    if y_pred.isna().any():
        raise ValueError(
            "Unexpected model prediction values: "
            f"{evaluation['ml_result'].unique().tolist()}"
        )

    # 6. Calculate evaluation metrics.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, pos_label="anomaly", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label="anomaly", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, pos_label="anomaly", zero_division=0
    )

    print("\n========== MODEL EVALUATION ==========")
    print(f"Total evaluated logins: {len(y_true)}")
    print(f"Actual anomalies:       {(y_true == 'anomaly').sum()}")
    print(f"Predicted anomalies:    {(y_pred == 'anomaly').sum()}")

    print("\n========== PERFORMANCE METRICS ==========")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n========== CONFUSION MATRIX ==========")
    print("Labels: normal, anomaly")
    print(confusion_matrix(
        y_true, y_pred, labels=["normal", "anomaly"]
    ))

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(
        y_true,
        y_pred,
        labels=["normal", "anomaly"],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    ))

    print("Evaluation completed.")


if __name__ == "__main__":
    main()
