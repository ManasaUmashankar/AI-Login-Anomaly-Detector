"""Compare unsupervised anomaly detectors on chronological holdout windows.

Labels are used only for scoring after prediction. All models are fit exclusively
on earlier rows. This is a synthetic-data comparison utility, not a benchmark.
"""
from pathlib import Path

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.anomaly_detector import MODEL_FEATURES, train_model, apply_model
from src.evaluate_model import (
    _choose_temporal_train_group,
    _temporal_group_starts,
    calculate_metrics,
    normalize_labels,
)
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "model_comparison.csv"


def _predictions_to_labels(predictions):
    """Translate sklearn's 1/-1 outlier convention to normal/anomaly."""
    return pd.Series(predictions).map({1: "normal", -1: "anomaly"}).reset_index(drop=True)


def evaluate_model_comparison(df=None, train_fraction=0.40, windows=4, seed=42):
    """Evaluate Isolation Forest, LOF novelty detection and One-Class SVM over time."""
    if df is None:
        df = load_logs(include_labels=True)
    if "true_label" not in df.columns:
        raise ValueError("Dataset must contain true_label for evaluation.")
    if not isinstance(windows, int) or windows < 2:
        raise ValueError("windows must be an integer of at least 2.")
    if not 0.30 <= train_fraction < 0.90:
        raise ValueError("train_fraction must be at least 0.30 and less than 0.90.")

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    working["true_label"] = normalize_labels(working["true_label"]).values
    cleaned = preprocess_logs(working, verbose=False)
    features = build_features(cleaned)

    starts = _temporal_group_starts(features)
    initial = _choose_temporal_train_group(features, train_fraction, windows)
    remaining = len(starts) - initial
    boundaries = [initial + (i * remaining) // windows for i in range(windows + 1)]
    records = []

    for window in range(windows):
        test_start = starts[boundaries[window]]
        test_end = starts[boundaries[window + 1]] if boundaries[window + 1] < len(starts) else len(features)
        train = features.iloc[:test_start]
        test = features.iloc[test_start:test_end]
        if len(train) < 3 or test.empty:
            raise ValueError("Each temporal split needs at least 3 training rows and 1 test row.")

        x_train = train[MODEL_FEATURES]
        x_test = test[MODEL_FEATURES]
        y_true = normalize_labels(test["true_label"])

        isolation_forest = train_model(train, seed=seed)
        isolation_result = apply_model(isolation_forest, test)
        predictions = {
            "Isolation Forest": isolation_result["ml_result"].map({
                "Normal": "normal", "Anomalous": "anomaly"
            }).reset_index(drop=True),
        }

        neighbors = min(20, len(train) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=neighbors,
            novelty=True,
            contamination=0.05,
        )
        lof.fit(x_train)
        predictions["Local Outlier Factor"] = _predictions_to_labels(lof.predict(x_test))

        svm = OneClassSVM(nu=0.05, gamma="scale")
        svm.fit(x_train)
        predictions["One-Class SVM"] = _predictions_to_labels(svm.predict(x_test))

        for model_name, y_pred in predictions.items():
            metrics = calculate_metrics(y_true, y_pred)
            records.append({
                "window": window + 1,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end": train["timestamp"].max(),
                "test_start": test["timestamp"].min(),
                "test_end": test["timestamp"].max(),
                **{key: metrics[key] for key in (
                    "precision", "recall", "f1", "false_positive_rate", "accuracy",
                    "tn", "fp", "fn", "tp",
                )},
            })

    return pd.DataFrame(records)


def main():
    results = evaluate_model_comparison()
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEFAULT_REPORT, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved comparison report: {DEFAULT_REPORT}")
    print("\nAll models were trained on earlier synthetic rows only. Labels were used only for scoring.")


if __name__ == "__main__":
    main()
