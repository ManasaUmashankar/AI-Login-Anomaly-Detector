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

CATEGORY_PRIOR_FIELDS = ["device", "ip_address", "location", "user_agent", "login_status"]
SUMMARY_METRICS = ("precision", "recall", "f1", "pr_auc")


def _metrics(y, pred):
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}


def summarize_seed_metrics(per_seed):
    """Summarize seed-level metrics with flat, unique CSV column names.

    Explicit named aggregation avoids pandas MultiIndex headers that create
    repeated metric names and a second header row when exported to CSV.
    """
    required = {"method", *SUMMARY_METRICS}
    missing = required.difference(per_seed.columns)
    if missing:
        raise ValueError(f"Missing columns required for robust summary: {sorted(missing)}")

    aggregations = {}
    for metric in SUMMARY_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    summary = per_seed.groupby("method", dropna=False).agg(**aggregations).round(4)
    # Sample standard deviation is undefined with one seed; keep it explicit.
    if "seed" in per_seed.columns and per_seed["seed"].nunique(dropna=True) <= 1:
        for metric in SUMMARY_METRICS:
            summary[f"{metric}_std"] = np.nan
    if not summary.columns.is_unique:
        raise ValueError("Robust evaluation summary contains duplicate column names.")
    return summary


def _supervised_category_prior(train, test, field):
    """One-field supervised baseline using training labels only.

    This is not an unsupervised anomaly detector: category anomaly rates are
    estimated from labelled training rows. Holdout labels are never consulted.
    """
    required = {field, "true_label"}
    if not required.issubset(train.columns) or field not in test.columns:
        raise ValueError(f"Training and test data must contain {field!r}; training data must contain true_label.")
    labels = train["true_label"].astype(str).str.strip().str.lower()
    if not labels.isin(["normal", "anomaly"]).all():
        raise ValueError("Training true_label must contain only normal/anomaly.")
    training = train[[field]].copy()
    training["true_label"] = labels.to_numpy()
    counts = training.groupby(field)["true_label"].agg(
        ["count", lambda s: (s == "anomaly").sum()]
    )
    counts.columns = ["n", "anomalies"]
    rates = (counts["anomalies"] + 1) / (counts["n"] + 2)
    threshold = float(labels.eq("anomaly").mean())
    return test[field].map(rates).fillna(0.0).gt(threshold).astype(int)


def hybrid_overlap_diagnostics(ml_predictions, rule_predictions):
    """Count how much the rule detector adds to the ML alert set."""
    ml = np.asarray(ml_predictions, dtype=int).reshape(-1)
    rules = np.asarray(rule_predictions, dtype=int).reshape(-1)
    if ml.shape != rules.shape:
        raise ValueError("ML and rule predictions must have the same length.")
    if not np.isin(ml, [0, 1]).all() or not np.isin(rules, [0, 1]).all():
        raise ValueError("ML and rule predictions must contain only binary 0/1 values.")

    ml_alerts = ml == 1
    rule_alerts = rules == 1
    overlap = ml_alerts & rule_alerts
    ml_exclusive = ml_alerts & ~rule_alerts
    rule_exclusive = rule_alerts & ~ml_alerts
    hybrid = ml_alerts | rule_alerts
    return {
        "rows": int(len(ml)),
        "ml_alerts": int(ml_alerts.sum()),
        "rule_alerts": int(rule_alerts.sum()),
        "overlap_alerts": int(overlap.sum()),
        "ml_exclusive_alerts": int(ml_exclusive.sum()),
        "rule_exclusive_alerts": int(rule_exclusive.sum()),
        "hybrid_alerts": int(hybrid.sum()),
        "rule_alerts_added_by_hybrid": int(rule_exclusive.sum()),
        "ml_alerts_added_by_hybrid": int(ml_exclusive.sum()),
        "predictions_identical": bool(np.array_equal(ml, rules)),
    }


def evaluate(seed, df, train_fraction=0.4, windows=4):
    clean = preprocess_logs(df.copy(), verbose=False).sort_values("timestamp", kind="stable").reset_index(drop=True)
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
        if end_time is not None:
            test = test[test["timestamp"] < end_time].copy()
        if train.empty or test.empty or test["true_label"].nunique() < 2:
            continue
        y = (test["true_label"] == "anomaly").astype(int)
        model = train_model(train, seed=seed, contamination="auto")
        score = -model.score_samples(test[MODEL_FEATURES])
        ml = (model.predict(test[MODEL_FEATURES]) == -1).astype(int)
        rule_result = apply_risk_scores(apply_rules(test.copy()))
        rules = (rule_result["risk_score"] >= 3).astype(int)
        rule_array = rules.to_numpy(dtype=int)
        overlap = hybrid_overlap_diagnostics(ml, rule_array)
        out.append({
            "seed": seed, "window": w + 1, "metric_type": "hybrid_overlap",
            "comparison": "ml_auto_vs_rules", **overlap,
        })
        hybrid = ((ml == 1) | (rule_array == 1)).astype(int)
        methods = {
            "supervised_category_prior_device": _supervised_category_prior(train, test, "device"),
            "supervised_category_prior_ip": _supervised_category_prior(train, test, "ip_address"),
            "supervised_category_prior_location": _supervised_category_prior(train, test, "location"),
            "supervised_category_prior_user_agent": _supervised_category_prior(train, test, "user_agent"),
            "supervised_category_prior_status": _supervised_category_prior(train, test, "login_status"),
            "rules_only": rules, "ml_only": ml, "hybrid": hybrid,
        }
        for contamination in (0.01, 0.03, 0.05, 0.10):
            sweep_model = train_model(train, seed=seed, contamination=contamination)
            sweep_predictions = (sweep_model.predict(test[MODEL_FEATURES]) == -1).astype(int)
            methods[f"ml_contamination_{contamination:.2f}"] = sweep_predictions
            if contamination in (0.03, 0.05):
                methods[f"hybrid_contamination_{contamination:.2f}_or_rules"] = ((sweep_predictions == 1) | (rule_array == 1)).astype(int)
                candidate_overlap = hybrid_overlap_diagnostics(sweep_predictions, rule_array)
                out.append({
                    "seed": seed, "window": w + 1, "metric_type": "hybrid_overlap",
                    "comparison": f"ml_contamination_{contamination:.2f}_vs_rules",
                    **candidate_overlap,
                })
        for method, pred in methods.items():
            m = _metrics(y, pred)
            row = {"seed": seed, "window": w + 1, "method": method, **m,
                   "pr_auc": float(average_precision_score(y, score)) if method == "ml_only" else np.nan}
            out.append(row)
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
                    out.append({
                        "seed": seed, "window": w + 1, "method": method,
                        "metric_type": "scenario", "scenario_id": sid,
                        "scenario_label": "attack" if not attack.empty else "benign",
                        "scenario_detected": int(not hit.empty) if not attack.empty else np.nan,
                        "scenario_false_positive": int(not false_alert.empty) if attack.empty else np.nan,
                        "time_to_detect_seconds": float((hit["timestamp"].min() - attack["timestamp"].min()).total_seconds()) if not hit.empty else np.nan,
                    })
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
    # First average chronological windows within each seed, then summarize the seed-level values.
    per_seed = event.groupby(["seed", "method"])[list(SUMMARY_METRICS)].mean().reset_index()
    summary = summarize_seed_metrics(per_seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(output, index=False, na_rep="NA")
    summary.to_csv(
        output.with_name(output.stem + "-summary.csv"), index_label="method", na_rep="NA"
    )
    print(summary.to_string())
    print(f"\nFull per-seed results: {output}")

    overlap_rows = all_runs[
        all_runs.get("metric_type", pd.Series(index=all_runs.index, dtype=object)) == "hybrid_overlap"
    ].copy()
    overlap_path = output.with_name(output.stem + "-overlap.csv")
    overlap_rows.to_csv(overlap_path, index=False, na_rep="NA")
    if not overlap_rows.empty:
        overlap_metrics = [
            "rows", "ml_alerts", "rule_alerts", "overlap_alerts",
            "ml_exclusive_alerts", "rule_exclusive_alerts", "hybrid_alerts",
            "rule_alerts_added_by_hybrid", "ml_alerts_added_by_hybrid",
        ]
        per_seed_overlap = overlap_rows.groupby(["seed", "comparison"], dropna=False)[overlap_metrics].mean()
        overlap_summary = per_seed_overlap.groupby("comparison").agg(["mean", "std"])
        overlap_summary.columns = [f"{metric}_{stat}" for metric, stat in overlap_summary.columns]
        overlap_summary = overlap_summary.rename(columns={"mean_mean": "mean_per_seed", "mean_std": "std_across_seeds"})
        overlap_summary_path = output.with_name(output.stem + "-overlap-summary.csv")
        overlap_summary.to_csv(overlap_summary_path, na_rep="NA")
        identical_rate = overlap_rows.groupby(["seed", "comparison"], dropna=False)["predictions_identical"].mean().groupby("comparison").mean()
        print("\nHybrid prediction overlap (window counts; mean across seeds):")
        print(overlap_summary.to_string())
        print("\nShare of windows with identical ML/rule predictions:")
        print(identical_rate.to_string())
        print(f"Overlap details: {overlap_path}")
        print(f"Overlap summary: {overlap_summary_path}")

    scenarios = all_runs[all_runs.get("metric_type", pd.Series(index=all_runs.index, dtype=object)) == "scenario"]
    if not scenarios.empty:
        attacks = scenarios[scenarios["scenario_label"] == "attack"]
        benign = scenarios[scenarios["scenario_label"] == "benign"]
        print("\nScenario-level attack detection rate / time-to-detect:")
        if not attacks.empty:
            print(attacks.groupby("method").agg(
                detection_rate=("scenario_detected", "mean"),
                mean_time_to_detect_seconds=("time_to_detect_seconds", "mean"),
            ).round(3).to_string())
        print("\nBenign scenario false-positive rate:")
        if not benign.empty:
            print(benign.groupby("method")["scenario_false_positive"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
