"""Repeatable temporal evaluation with event and scenario-level metrics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from src.anomaly_detector import MODEL_FEATURES, train_model
from src.detection_rules import apply_risk_scores, apply_rules
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

TRIVIAL_FIELDS = ["device", "ip_address", "location", "user_agent", "login_status"]

def _metrics(y, pred):
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}

def _single_field(train, test, field):
    # Smoothed train-only category anomaly rate; no holdout labels used to choose categories.
    counts = train.groupby(field)["true_label"].agg(["count", lambda s: (s == "anomaly").sum()])
    counts.columns = ["n", "anomalies"]
    rates = (counts["anomalies"] + 1) / (counts["n"] + 2)
    threshold = float((train["true_label"] == "anomaly").mean())
    return test[field].map(rates).fillna(threshold).ge(threshold).astype(int)

def evaluate(seed, df, train_fraction=0.4, windows=4):
    clean = preprocess_logs(df.copy()).sort_values("timestamp", kind="stable").reset_index(drop=True)
    clean["true_label"] = clean["true_label"].astype(str).str.lower()
    features = build_features(clean)
    timestamps = features["timestamp"]
    unique_times = timestamps.drop_duplicates().tolist()
    first = max(2, int(len(unique_times) * train_fraction))
    boundaries = np.linspace(first, len(unique_times), windows + 1, dtype=int)
    out = []
    for w in range(windows):
        start_time = unique_times[boundaries[w]] if boundaries[w] < len(unique_times) else unique_times[-1]
        end_time = unique_times[boundaries[w+1]] if boundaries[w+1] < len(unique_times) else None
        train = features[features["timestamp"] < start_time].copy()
        test = features[features["timestamp"] >= start_time].copy()
        if end_time is not None: test = test[test["timestamp"] < end_time].copy()
        if train.empty or test.empty or test["true_label"].nunique() < 2: continue
        y = (test["true_label"] == "anomaly").astype(int)
        model = train_model(train, seed=seed, contamination="auto")
        score = -model.score_samples(test[MODEL_FEATURES])
        ml = (model.predict(test[MODEL_FEATURES]) == -1).astype(int)
        rule_result = apply_risk_scores(apply_rules(test.copy()))
        rules = (rule_result["risk_score"] >= 3).astype(int)
        hybrid = ((ml == 1) | (rules.to_numpy() == 1)).astype(int)
        methods = {"single_field_device": _single_field(train, test, "device"), "single_field_ip": _single_field(train, test, "ip_address"), "single_field_location": _single_field(train, test, "location"), "single_field_user_agent": _single_field(train, test, "user_agent"), "single_field_status": _single_field(train, test, "login_status"), "rules_only": rules, "ml_only": ml, "hybrid": hybrid}
        for contamination in (0.01, 0.03, 0.05, 0.10):
            sweep_model = train_model(train, seed=seed, contamination=contamination)
            methods[f"ml_contamination_{contamination:.2f}"] = (sweep_model.predict(test[MODEL_FEATURES]) == -1).astype(int)
        for method, pred in methods.items():
            m = _metrics(y, pred)
            row = {"seed": seed, "window": w + 1, "method": method, **m, "pr_auc": float(average_precision_score(y, score)) if method == "ml_only" else np.nan}
            out.append(row)
        # Keep attack and benign scenarios; exclude ambiguous labels rather than guessing.
        if "scenario_id" in test:
            for method, pred in methods.items():
                temp = test[["scenario_id", "timestamp", "true_label"]].copy()
                temp["pred"] = np.asarray(pred)
                temp = temp[(temp["scenario_id"].fillna("") != "") & temp["true_label"].isin(["normal", "anomaly"])]
                for sid, grp in temp.groupby("scenario_id"):
                    attack = grp[grp["true_label"] == "anomaly"]
                    benign = grp[grp["true_label"] == "normal"]
                    hit = attack[attack["pred"] == 1]
                    false_alert = benign[benign["pred"] == 1]
                    out.append({"seed": seed, "window": w + 1, "method": method, "metric_type": "scenario", "scenario_id": sid, "scenario_label": "attack" if not attack.empty else "benign", "scenario_detected": int(not hit.empty) if not attack.empty else np.nan, "scenario_false_positive": int(not false_alert.empty) if attack.empty else np.nan, "time_to_detect_seconds": float((hit["timestamp"].min() - attack["timestamp"].min()).total_seconds()) if not hit.empty else np.nan})
    return pd.DataFrame(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--output", default="reports/robust-evaluation.csv")
    args = parser.parse_args()
    data = load_logs(include_labels=True)
    all_runs = pd.concat([evaluate(seed, data, windows=args.windows) for seed in range(args.seeds)], ignore_index=True)
    event = all_runs[all_runs.get("metric_type", pd.Series(index=all_runs.index, dtype=object)).isna()]
    summary = event.groupby("method")[["precision", "recall", "f1", "pr_auc"]].agg(["mean", "std"]).round(4)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(output, index=False)
    summary.to_csv(output.with_name(output.stem + "-summary.csv"))
    print(summary.to_string()); print(f"\nFull per-seed results: {output}")
    scenarios = all_runs[all_runs.get("metric_type", pd.Series(index=all_runs.index, dtype=object)) == "scenario"]
    if not scenarios.empty: print("\nScenario detection rate / time-to-detect:"); print(scenarios.groupby("method").agg(detection_rate=("scenario_detected", "mean"), mean_time_to_detect_seconds=("time_to_detect_seconds", "mean")).round(3).to_string())

if __name__ == "__main__": main()
