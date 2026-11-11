"""Evaluate the detector on an independently labelled real authentication CSV.

No synthetic scenarios are generated here. Labels are used only after the
chronological split for scoring; the model is fitted on earlier records only.
"""
import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd

from src.anomaly_detector import apply_model, train_model
from src.detection_rules import apply_risk_scores, apply_rules
from src.evaluate_model import (
    _choose_temporal_train_group,
    _temporal_group_starts,
    calculate_metrics,
    normalize_labels,
)
from src.feature_engineering import build_features
from src.preprocessing import REQUIRED_COLUMNS, preprocess_logs


def evaluate_real_world_csv(csv_path, *, label_provenance, train_fraction=0.70):
    """Return metrics and a compact report for an external labelled CSV."""
    if not label_provenance or len(label_provenance.strip()) < 12:
        raise ValueError("Describe the independent label source with --label-provenance.")
    provenance_lower = label_provenance.lower()
    prohibited_claims = ("synthetic", "same generator", "generated labels")
    if any(term in provenance_lower for term in prohibited_claims):
        raise ValueError(
            "The label source must be independent; synthetic or generator-derived labels "
            "cannot substantiate real-world evaluation."
        )
    source = Path(csv_path)
    frame = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS + ["true_label"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Real-world CSV is missing required columns: {missing}")
    if not 0.50 <= train_fraction < 0.90:
        raise ValueError("train_fraction must be between 0.50 and 0.90.")

    labels = normalize_labels(frame["true_label"])
    frame["true_label"] = labels
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if working.empty:
        raise ValueError("No rows with valid timestamps were found.")
    cleaned = preprocess_logs(working, verbose=False)
    if len(cleaned) < 10:
        raise ValueError("At least 10 valid labelled events are required for evaluation.")

    features = build_features(cleaned)
    starts = _temporal_group_starts(features)
    split_group = _choose_temporal_train_group(
        features, train_fraction, minimum_future_groups=1
    )
    split_at = starts[split_group]
    train = features.iloc[:split_at].copy()
    test = features.iloc[split_at:].copy()
    if test.empty:
        raise ValueError("The chronological holdout is empty.")

    # Labels are not included in model training or prediction features.
    model = train_model(train.drop(columns=["true_label"], errors="ignore"))
    ml = apply_model(model, test.drop(columns=["true_label"], errors="ignore"))
    ml_pred = ml["ml_result"].map({"Normal": "normal", "Anomalous": "anomaly"})
    rule_result = apply_risk_scores(apply_rules(test.drop(columns=["true_label"], errors="ignore")))
    rule_pred = rule_result["risk_score"].ge(3).map({True: "anomaly", False: "normal"})
    hybrid_pred = (
        ml_pred.eq("anomaly") | rule_pred.eq("anomaly")
    ).map({True: "anomaly", False: "normal"})
    actual = normalize_labels(test["true_label"])
    holdout_label_counts = actual.value_counts().to_dict()
    if set(holdout_label_counts) != {"normal", "anomaly"} or min(holdout_label_counts.values()) < 2:
        raise ValueError(
            "Chronological holdout must contain at least two independently labelled "
            "normal events and two anomaly events; provide a larger or better-covered dataset."
        )

    dataset_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    metrics = {
        "isolation_forest": calculate_metrics(actual, ml_pred),
        "rules": calculate_metrics(actual, rule_pred),
        "hybrid": calculate_metrics(actual, hybrid_pred),
    }
    report = {
        "evaluation_type": "independently_labelled_external_authentication_logs",
        "dataset_file_name": source.name,
        "dataset_sha256": dataset_sha256,
        "label_provenance": label_provenance.strip(),
        "rows_after_preprocessing": len(features),
        "training_rows": len(train),
        "holdout_rows": len(test),
        "training_end": str(train["timestamp"].max()),
        "holdout_start": str(test["timestamp"].min()),
        "holdout_end": str(test["timestamp"].max()),
        "holdout_label_counts": {key: int(value) for key, value in holdout_label_counts.items()},
        "actual_holdout_anomalies": int((actual == "anomaly").sum()),
        "metrics": {
            name: {
                key: float(value) if hasattr(value, "item") else value
                for key, value in result.items()
                if key not in {"confusion_matrix"}
            } | {"confusion_matrix": result["confusion_matrix"].tolist()}
            for name, result in metrics.items()
        },
        "limitations": [
            "Results depend on independent label quality and representativeness.",
            "The synthetic-trained model is not assumed to generalise to this environment.",
            "Use a separate representative training baseline before operational deployment.",
            "Review false positives and false negatives with a security analyst.",
            "A small holdout is a smoke test only; confidence intervals and broader time/environment coverage are still needed.",
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to a real authentication CSV; not uploaded by this script.")
    parser.add_argument(
        "--label-provenance",
        required=True,
        help="How normal/anomaly labels were independently established.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--output", default="reports/real-world-evaluation.json")
    args = parser.parse_args()

    report = evaluate_real_world_csv(
        args.csv_path,
        label_provenance=args.label_provenance,
        train_fraction=args.train_fraction,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
