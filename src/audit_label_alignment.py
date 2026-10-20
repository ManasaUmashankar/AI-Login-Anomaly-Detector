"""Audit alignment between synthetic ground-truth labels and rule-based signals."""
import pandas as pd

from src.detection_rules import RISK_SIGNALS, apply_risk_scores, apply_rules
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs


def binary_metrics(y_true, y_pred):
    """Calculate binary metrics with anomaly as the positive class."""
    actual = pd.Series(y_true).astype(str).str.strip().str.lower().reset_index(drop=True)
    predicted = pd.Series(y_pred).astype(str).str.strip().str.lower().reset_index(drop=True)
    valid = {"normal", "anomaly"}
    if len(actual) == 0:
        raise ValueError("Cannot audit an empty dataset.")
    if len(actual) != len(predicted):
        raise ValueError("Actual labels and predictions must have equal lengths.")
    if not set(actual.unique()).issubset(valid):
        raise ValueError("Labels must contain only 'normal' and 'anomaly'.")
    if not set(predicted.unique()).issubset(valid):
        raise ValueError("Predictions must contain only 'normal' and 'anomaly'.")

    actual_anomaly = actual.eq("anomaly")
    predicted_anomaly = predicted.eq("anomaly")
    tp = int((actual_anomaly & predicted_anomaly).sum())
    fp = int((~actual_anomaly & predicted_anomaly).sum())
    fn = int((actual_anomaly & ~predicted_anomaly).sum())
    tn = int((~actual_anomaly & ~predicted_anomaly).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(actual),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def audit_label_alignment(df, risk_threshold=3):
    """Compare individual rule signals and combined risk scores with synthetic labels."""
    if not isinstance(risk_threshold, int) or isinstance(risk_threshold, bool) or risk_threshold < 0:
        raise ValueError("risk_threshold must be a non-negative integer.")
    if "true_label" not in df.columns:
        raise ValueError("Dataset must contain a true_label column for auditing.")

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    working["true_label"] = working["true_label"].astype(str).str.strip().str.lower()
    if not working["true_label"].isin(["normal", "anomaly"]).all():
        raise ValueError("true_label must contain only 'normal' and 'anomaly'.")

    cleaned = preprocess_logs(working)
    features = build_features(cleaned)
    actual = features["true_label"].reset_index(drop=True)
    risk = apply_risk_scores(apply_rules(features.copy()))

    detector_results = []
    for name, _points, signal_function in RISK_SIGNALS:
        prediction = signal_function(features).map({True: "anomaly", False: "normal"})
        detector_results.append({"detector": name, **binary_metrics(actual, prediction)})

    baseline_prediction = risk["risk_score"].ge(risk_threshold).map(
        {True: "anomaly", False: "normal"}
    )
    detector_results.append({
        "detector": f"Combined risk score >= {risk_threshold}",
        **binary_metrics(actual, baseline_prediction),
    })

    score_table = risk.assign(true_label=actual).groupby("risk_score").agg(
        rows=("true_label", "size"),
        anomalies=("true_label", lambda s: int(s.eq("anomaly").sum())),
        anomaly_rate=("true_label", lambda s: float(s.eq("anomaly").mean())),
    ).reset_index()

    return {
        "rows_audited": len(features),
        "label_counts": actual.value_counts().to_dict(),
        "detectors": detector_results,
        "risk_score_distribution": score_table.to_dict(orient="records"),
        "risk_threshold": risk_threshold,
    }


def main():
    report = audit_label_alignment(load_logs(include_labels=True))
    print("\n========== SYNTHETIC LABEL / RULE ALIGNMENT AUDIT ==========")
    print(f"Rows audited: {report['rows_audited']}")
    print(f"Ground-truth label counts: {report['label_counts']}")
    print("\nAgreement with synthetic labels:")
    for item in report["detectors"]:
        print(
            f"- {item['detector']}: precision={item['precision']:.3f}, "
            f"recall={item['recall']:.3f}, F1={item['f1']:.3f}, "
            f"FPR={item['false_positive_rate']:.3f}, accuracy={item['accuracy']:.3f}"
        )
    print("\nAnomaly rate by combined risk score:")
    for item in report["risk_score_distribution"]:
        print(
            f"- score {item['risk_score']}: {item['anomalies']}/{item['rows']} "
            f"labelled anomalies (rate={item['anomaly_rate']:.3f})"
        )
    print(
        "\nCaution: this measures agreement with labels in synthetic data. "
        "If the generator encodes these same signals into scenarios, high scores "
        "are expected and are not independent evidence of real-world SOC effectiveness."
    )


if __name__ == "__main__":
    main()
