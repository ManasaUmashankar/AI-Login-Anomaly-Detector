"""Evaluate ML and rule detectors on deliberately difficult synthetic login scenarios.

This stress test is separate from the original generated labels. It is a controlled
engineering diagnostic, not a substitute for evaluation on real authentication logs.
"""
import pandas as pd

from src.anomaly_detector import apply_model, train_model
from src.detection_rules import apply_risk_scores, apply_rules
from src.evaluate_model import calculate_metrics
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

STRESS_COLUMNS = [
    "timestamp", "username", "ip_address", "location", "device",
    "login_status", "failed_attempts", "user_agent", "session_id", "true_label",
]


def _user_profiles(history):
    profiles = {}
    for username, group in history.groupby("username", sort=True):
        profile = {"username": username}
        for column in ("ip_address", "location", "device"):
            modes = group[column].mode(dropna=True)
            profile[column] = modes.iloc[0] if not modes.empty else "unknown"
        profiles[username] = profile
    if not profiles:
        raise ValueError("Historical login data is required to build stress scenarios.")
    return profiles


def build_stress_scenarios(history):
    """Build benign edge cases and suspicious sequences after historical data."""
    required = {"timestamp", "username", "ip_address", "location", "device",
                "login_status", "failed_attempts", "user_agent", "session_id"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Historical data is missing required columns: {missing}")

    clean_history = history.copy()
    clean_history["timestamp"] = pd.to_datetime(clean_history["timestamp"], errors="coerce")
    clean_history = clean_history.dropna(subset=["timestamp"])
    if clean_history.empty:
        raise ValueError("Historical login data must contain valid timestamps.")

    profiles = _user_profiles(clean_history)
    usernames = list(profiles)
    start = clean_history["timestamp"].max().normalize() + pd.Timedelta(days=1, hours=10)
    rows = []
    counter = 0

    def emit(profile, *, minutes=0, seconds=0, ip=None, location=None,
             device=None, status="success", failures=0, label="normal",
             scenario=""):
        nonlocal counter
        counter += 1
        actual_device = device or profile["device"]
        rows.append({
            "timestamp": start + pd.Timedelta(minutes=minutes, seconds=seconds),
            "username": profile["username"],
            "ip_address": ip or profile["ip_address"],
            "location": location or profile["location"],
            "device": actual_device,
            "login_status": status,
            "failed_attempts": failures,
            "user_agent": {
                "Windows Laptop": "Chrome on Windows",
                "MacBook": "Safari on Mac",
                "Android Phone": "Chrome on Android",
                "Unknown Device": "python-requests",
            }.get(actual_device, "Synthetic stress-test agent"),
            "session_id": f"stress-{counter:04d}",
            "true_label": label,
            "scenario": scenario,
        })

    p0 = profiles[usernames[0]]
    p1 = profiles[usernames[min(1, len(usernames) - 1)]]
    p2 = profiles[usernames[min(2, len(usernames) - 1)]]
    p3 = profiles[usernames[min(3, len(usernames) - 1)]]

    # Benign edge cases: a single changed attribute should not automatically mean compromise.
    emit(p0, minutes=1, ip="198.51.100.201", scenario="benign_new_ip_only")
    emit(p1, minutes=2, device="Temporary Work Laptop", scenario="benign_new_device_only")
    emit(p2, minutes=3, location="Jaipur, India", scenario="benign_new_location_only")
    emit(p3, minutes=-420, scenario="benign_off_hours_only")

    # A legitimate user may have several successful logins over a longer period.
    # Six-minute gaps should not trigger the five-minute rapid-success rule.
    for attempt in range(1, 7):
        emit(
            p0, minutes=100 + attempt * 6, status="success",
            scenario="benign_successes_spread_out",
        )

    # Ambiguous cases resemble suspicious patterns, so do not force a benign/attack label.
    # Their alerts are reported separately and excluded from binary performance metrics.
    for attempt in range(6):
        emit(
            p0, minutes=150, seconds=attempt * 8, status="success",
            label="ambiguous", scenario="ambiguous_rapid_success_familiar_context",
        )
    for attempt in range(6):
        emit(
            p1, minutes=160 + attempt * 7, status="failed", failures=attempt + 1,
            label="ambiguous", scenario="ambiguous_low_and_slow_failures",
        )
    emit(
        p3, minutes=220, device="Temporary Work Laptop", location="Jaipur, India",
        label="ambiguous", scenario="ambiguous_new_device_and_location_daytime",
    )

    # Suspicious activity with otherwise familiar context.
    for attempt in range(1, 7):
        emit(
            p0, minutes=10, seconds=attempt * 8, status="failed",
            failures=attempt, label="anomaly", scenario="anomaly_brute_force_familiar_context",
        )
    for attempt in range(6):
        emit(
            p1, minutes=20, seconds=attempt * 8, status="success",
            failures=0, label="anomaly", scenario="anomaly_rapid_success_familiar_context",
        )

    # Low-and-slow sequence: repeated failures without crossing the five-minute burst threshold.
    for attempt in range(6):
        emit(
            p2, minutes=30 + attempt * 7, status="failed", failures=attempt + 1,
            label="anomaly", scenario="anomaly_low_and_slow_failures",
        )

    # A new IP at a normal hour, paired with a new location: suspicious but not the generator's
    # original three-signal pattern (new IP + new device + unusual location + unusual hour).
    emit(
        p3, minutes=80, ip="198.51.100.202", location="Jaipur, India",
        label="anomaly", scenario="anomaly_new_ip_and_location_daytime",
    )

    # A new device and location during daytime, without changing the IP.
    emit(
        p3, minutes=90, device="Temporary Work Laptop", location="Jaipur, India",
        label="anomaly", scenario="anomaly_new_device_and_location_daytime",
    )

    # Password-spraying-style sequence: one source IP fails against several accounts.
    spray_profiles = [p0, p1, p2, p3]
    for attempt, profile in enumerate(spray_profiles, start=1):
        emit(
            profile, minutes=240, seconds=attempt * 20,
            ip="198.51.100.203", status="failed", failures=1,
            label="anomaly", scenario="anomaly_cross_account_password_spray",
        )

    # Shared office/NAT networks can produce a similar pattern without malicious intent.
    # Keep this case ambiguous so the rule is reviewed, not scored as definitely malicious.
    for attempt, profile in enumerate(spray_profiles, start=1):
        emit(
            profile, minutes=250, seconds=attempt * 20,
            ip="198.51.100.204", status="failed", failures=1,
            label="ambiguous", scenario="ambiguous_shared_network_failures",
        )

    result = pd.DataFrame(rows, columns=STRESS_COLUMNS + ["scenario"])
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def evaluate_stress_test(history=None):
    """Fit the unsupervised model on history, then score later labelled stress scenarios."""
    if history is None:
        history = load_logs(include_labels=True)
    if "true_label" not in history.columns:
        raise ValueError("Historical data must contain true_label for evaluation.")

    historical = history.copy()
    historical["timestamp"] = pd.to_datetime(historical["timestamp"], errors="coerce")
    historical = historical.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    historical = preprocess_logs(historical, verbose=False)
    historical_features = build_features(historical)

    stress = build_stress_scenarios(historical_features)
    combined = pd.concat([historical_features, stress], ignore_index=True, sort=False)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    all_features = build_features(combined)
    stress_features = all_features[all_features["scenario"].notna()].copy().reset_index(drop=True)

    # The labels are never passed into train_model; Isolation Forest remains unsupervised.
    model = train_model(historical_features)
    ml = apply_model(model, stress_features)
    rules = apply_risk_scores(apply_rules(stress_features.copy()))
    actual = stress_features["true_label"].reset_index(drop=True)
    ml_prediction = ml["ml_result"].map({"Normal": "normal", "Anomalous": "anomaly"})
    rule_prediction = rules["risk_score"].ge(3).map({True: "anomaly", False: "normal"})
    # Hybrid alerts if either detector raises an alert; this is a layered detector, not an ML-only result.
    hybrid_prediction = (ml_prediction.eq("anomaly") | rule_prediction.eq("anomaly")).map(
        {True: "anomaly", False: "normal"}
    )

    # Score only clearly labelled cases; retain ambiguous cases for separate alert review.
    known_mask = actual.isin(["normal", "anomaly"])
    if not known_mask.any():
        raise ValueError("Stress evaluation requires clearly labelled scenario events.")
    summary = {
        "ml": calculate_metrics(actual[known_mask], ml_prediction[known_mask]),
        "rules": calculate_metrics(actual[known_mask], rule_prediction[known_mask]),
        "hybrid": calculate_metrics(actual[known_mask], hybrid_prediction[known_mask]),
        "evaluated_events": int(known_mask.sum()),
        "ambiguous_events": int((~known_mask).sum()),
    }
    diagnostic_columns = [
        "scenario", "username", "timestamp", "true_label",
        "failed_logins_45m", "successful_logins_5m",
        "distinct_failed_users_per_ip_15m", "login_frequency",
        "is_new_ip", "is_new_device", "is_unusual_location", "location_change",
    ]
    rows = stress_features[diagnostic_columns].copy()
    rows["risk_score"] = rules["risk_score"].to_numpy()
    rows["rule_prediction"] = rule_prediction.to_numpy()
    rows["ml_prediction"] = ml_prediction.to_numpy()
    rows["hybrid_prediction"] = hybrid_prediction.to_numpy()
    rows["ml_anomaly_score"] = ml["anomaly_score"].to_numpy()
    known_rows = rows["true_label"].isin(["normal", "anomaly"])
    rows["ml_correct"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    rows["rules_correct"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    rows.loc[known_rows, "ml_correct"] = rows.loc[known_rows, "ml_prediction"].eq(rows.loc[known_rows, "true_label"])
    rows.loc[known_rows, "rules_correct"] = rows.loc[known_rows, "rule_prediction"].eq(rows.loc[known_rows, "true_label"])
    return summary, rows


def evaluate_scenario_detection(rows):
    """Summarize whether each complete scenario generated an alert and how quickly.

    Input rows must be in chronological order within each scenario. Detection delay
    is the 1-based event position of the first alert; None means no alert occurred.
    Scenario-level metrics count each scenario once, avoiding treating every early
    event in a known suspicious sequence as a separate independent scenario.
    """
    required = {"scenario", "true_label", "rule_prediction", "ml_prediction"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Stress results are missing required columns: {missing}")
    if rows.empty:
        raise ValueError("Stress results must not be empty.")

    scenario_rows = []
    for scenario, group in rows.groupby("scenario", sort=False):
        group = group.reset_index(drop=True)
        row = {
            "scenario": scenario,
            "true_label": group["true_label"].iloc[0],
            "evaluation_scope": "ambiguous_review" if group["true_label"].iloc[0] == "ambiguous" else "scored",
            "events": len(group),
        }
        detector_columns = [("rule", "rule_prediction"), ("ml", "ml_prediction")]
        if "hybrid_prediction" in group.columns:
            detector_columns.append(("hybrid", "hybrid_prediction"))
        for detector, prediction_column in detector_columns:
            alerts = group[prediction_column].eq("anomaly").tolist()
            first_alert = next((index + 1 for index, alert in enumerate(alerts) if alert), None)
            row[f"{detector}_detected"] = first_alert is not None
            row[f"{detector}_detection_delay_events"] = first_alert
        scenario_rows.append(row)

    scenario_summary = pd.DataFrame(scenario_rows)
    known_mask = scenario_summary["true_label"].isin(["normal", "anomaly"])
    metrics = {}
    for detector in ("rule", "ml", "hybrid"):
        if f"{detector}_detected" not in scenario_summary.columns:
            continue
        predicted = scenario_summary[f"{detector}_detected"].map(
            {True: "anomaly", False: "normal"}
        )
        metrics[detector] = calculate_metrics(
            scenario_summary.loc[known_mask, "true_label"],
            predicted[known_mask],
        )
    return metrics, scenario_summary


def _print_metrics(name, metrics):
    print(
        f"{name:12s} precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
        f"FPR={metrics['false_positive_rate']:.3f} "
        f"accuracy={metrics['accuracy']:.3f} "
        f"(TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']})"
    )


def main():
    summary, rows = evaluate_stress_test()
    print("\n========== SYNTHETIC EDGE-CASE STRESS TEST ==========")
    print("Model trained on historical rows only; stress scenarios are held out.")
    print("All cases are synthetic and deliberately selected, not a real-world benchmark.")
    print(f"Clearly labelled events scored: {summary['evaluated_events']}; ambiguous events excluded from binary metrics: {summary['ambiguous_events']}.\n")
    _print_metrics("ML model", summary["ml"])
    _print_metrics("Rule baseline", summary["rules"])
    _print_metrics("Hybrid detector", summary["hybrid"])
    scenario_metrics, scenario_rows = evaluate_scenario_detection(rows)
    print("\n========== SCENARIO-LEVEL DETECTION ==========")
    print("Each complete sequence counts once; delay is the event position of the first alert.")
    _print_metrics("ML scenarios", scenario_metrics["ml"])
    _print_metrics("Rule scenarios", scenario_metrics["rule"])
    _print_metrics("Hybrid scenarios", scenario_metrics["hybrid"])
    print("\nScenario outcomes and detection delay:")
    display_rows = scenario_rows.copy()
    for detector in ("rule", "ml", "hybrid"):
        display_rows[f"{detector}_detection_delay_events"] = display_rows[
            f"{detector}_detection_delay_events"
        ].map(lambda value: value if pd.notna(value) else "Not detected")
    print(display_rows.to_string(index=False))
    ambiguous_rows = display_rows[display_rows["evaluation_scope"] == "ambiguous_review"]
    print("\nAmbiguous-case alert review (not counted as false positives or true positives):")
    if ambiguous_rows.empty:
        print("No ambiguous scenarios.")
    else:
        print(ambiguous_rows[["scenario", "events", "rule_detected", "ml_detected", "hybrid_detected"]].to_string(index=False))

    print("\nPer-event outcomes:")
    display_events = rows[[
        "scenario", "true_label", "failed_logins_45m", "successful_logins_5m",
        "distinct_failed_users_per_ip_15m", "login_frequency", "is_new_ip", "is_new_device", "is_unusual_location",
        "location_change", "risk_score", "ml_anomaly_score", "rule_prediction",
        "ml_prediction", "hybrid_prediction", "rules_correct", "ml_correct",
    ]].copy()
    display_events["ml_anomaly_score"] = display_events["ml_anomaly_score"].round(4)
    print(display_events.to_string(index=False))
    print(
        "\nInterpretation: binary metrics use only clearly labelled cases. Ambiguous cases "
        "are reported separately because a pattern alone cannot establish malicious intent. "
        "The stress set is intentionally small and is not a real-world benchmark."
    )


if __name__ == "__main__":
    main()
