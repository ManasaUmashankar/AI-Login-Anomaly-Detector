import pandas as pd

from src.preprocessing import REQUIRED_COLUMNS
from src.stress_test_evaluation import (
    build_stress_scenarios,
    build_error_audit,
    evaluate_scenario_detection,
    evaluate_stress_test,
)


def make_history():
    rows = []
    profiles = [
        ("alice", "203.0.113.11", "Mumbai, India", "Windows Laptop"),
        ("bob", "203.0.113.12", "Pune, India", "MacBook"),
        ("cara", "203.0.113.13", "Bengaluru, India", "Android Phone"),
        ("dan", "203.0.113.14", "Chennai, India", "Windows Laptop"),
    ]
    for day in range(1, 11):
        for username, ip, location, device in profiles:
            rows.append([
                f"2026-09-{day:02d} 10:00:00", username, ip, location, device,
                "success", 0, "test-agent", f"{username}-{day}",
            ])
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def test_stress_scenarios_keep_benign_and_anomaly_labels_separate():
    stress = build_stress_scenarios(make_history())

    assert len(stress) == 62
    assert set(stress["true_label"]) == {"normal", "anomaly", "ambiguous"}
    benign = stress[stress["scenario"].str.startswith("benign_")]
    ambiguous = stress[stress["scenario"].str.startswith("ambiguous_")]
    suspicious = stress[stress["scenario"].str.startswith("anomaly_")]
    assert len(benign) == 21
    assert len(ambiguous) == 17
    assert len(suspicious) == 24
    assert benign["true_label"].eq("normal").all()
    assert ambiguous["true_label"].eq("ambiguous").all()
    assert suspicious["true_label"].eq("anomaly").all()


def test_stress_set_includes_single_signal_benign_edge_cases():
    stress = build_stress_scenarios(make_history()).set_index("scenario")

    assert stress.loc["benign_new_ip_only", "ip_address"] == "198.51.100.201"
    assert stress.loc["benign_new_device_only", "device"] == "Temporary Work Laptop"
    assert stress.loc["benign_new_location_only", "location"] == "Jaipur, India"
    assert stress.loc["benign_off_hours_only", "timestamp"].hour == 3


def test_suspicious_scenarios_include_familiar_context_and_low_and_slow():
    stress = build_stress_scenarios(make_history())

    brute_force = stress[stress["scenario"] == "anomaly_brute_force_familiar_context"]
    low_slow = stress[stress["scenario"] == "anomaly_low_and_slow_failures"]
    rapid = stress[stress["scenario"] == "anomaly_rapid_success_familiar_context"]

    assert len(brute_force) == 6
    assert len(low_slow) == 6
    assert len(rapid) == 6
    assert brute_force["ip_address"].nunique() == 1
    assert rapid["login_status"].eq("success").all()
    assert low_slow["timestamp"].diff().dropna().dt.total_seconds().eq(420).all()

def test_stress_set_includes_spaced_benign_successes_and_daytime_contextual_anomaly():
    stress = build_stress_scenarios(make_history())

    spaced = stress[stress["scenario"] == "benign_successes_spread_out"]
    contextual = stress[stress["scenario"] == "anomaly_new_device_and_location_daytime"]

    assert len(spaced) == 6
    assert spaced["true_label"].eq("normal").all()
    assert spaced["login_status"].eq("success").all()
    assert spaced["timestamp"].diff().dropna().dt.total_seconds().eq(360).all()
    assert len(contextual) == 1
    assert contextual.iloc[0]["true_label"] == "anomaly"
    assert contextual.iloc[0]["device"] == "Temporary Work Laptop"
    assert contextual.iloc[0]["location"] == "Jaipur, India"



def test_stress_set_includes_hard_negative_benign_scenarios():
    stress = build_stress_scenarios(make_history())

    rapid = stress[stress["scenario"] == "ambiguous_rapid_success_familiar_context"]
    low_slow = stress[stress["scenario"] == "ambiguous_low_and_slow_failures"]
    contextual = stress[stress["scenario"] == "ambiguous_new_device_and_location_daytime"]

    assert len(rapid) == 6
    assert rapid["login_status"].eq("success").all()
    assert rapid["true_label"].eq("ambiguous").all()
    assert len(low_slow) == 6
    assert low_slow["login_status"].eq("failed").all()
    assert low_slow["true_label"].eq("ambiguous").all()
    assert len(contextual) == 1
    assert contextual.iloc[0]["true_label"] == "ambiguous"


def test_stress_set_includes_cross_account_spray_and_shared_network_ambiguity():
    stress = build_stress_scenarios(make_history())

    spray = stress[stress["scenario"] == "anomaly_cross_account_password_spray"]
    shared = stress[stress["scenario"] == "ambiguous_shared_network_failures"]

    assert len(spray) == 4
    assert spray["username"].nunique() == 4
    assert spray["ip_address"].nunique() == 1
    assert spray["login_status"].eq("failed").all()
    assert spray["true_label"].eq("anomaly").all()

    assert len(shared) == 4
    assert shared["username"].nunique() == 4
    assert shared["ip_address"].nunique() == 1
    assert shared["true_label"].eq("ambiguous").all()


def test_scenario_evaluation_reports_detection_and_first_alert_position():
    rows = pd.DataFrame([
        {"scenario": "benign", "true_label": "normal",
         "rule_prediction": "normal", "ml_prediction": "normal"},
        {"scenario": "slow_attack", "true_label": "anomaly",
         "rule_prediction": "normal", "ml_prediction": "normal"},
        {"scenario": "slow_attack", "true_label": "anomaly",
         "rule_prediction": "anomaly", "ml_prediction": "normal"},
        {"scenario": "single_attack", "true_label": "anomaly",
         "rule_prediction": "anomaly", "ml_prediction": "anomaly"},
    ])

    metrics, scenarios = evaluate_scenario_detection(rows)
    by_name = scenarios.set_index("scenario")

    assert by_name.loc["slow_attack", "rule_detection_delay_events"] == 2
    assert pd.isna(by_name.loc["slow_attack", "ml_detection_delay_events"])
    assert by_name.loc["single_attack", "rule_detection_delay_events"] == 1
    assert metrics["rule"]["recall"] == 1.0
    assert metrics["ml"]["recall"] == 0.5
    assert metrics["rule"]["false_positive_rate"] == 0.0
    assert metrics["ml"]["false_positive_rate"] == 0.0



def test_ambiguous_scenarios_are_excluded_from_binary_metrics_but_reported():
    rows = pd.DataFrame([
        {"scenario": "clear_benign", "true_label": "normal",
         "rule_prediction": "normal", "ml_prediction": "normal"},
        {"scenario": "clear_attack", "true_label": "anomaly",
         "rule_prediction": "anomaly", "ml_prediction": "normal"},
        {"scenario": "uncertain", "true_label": "ambiguous",
         "rule_prediction": "anomaly", "ml_prediction": "normal"},
    ])

    metrics, scenarios = evaluate_scenario_detection(rows)
    assert metrics["rule"]["tp"] == 1
    assert metrics["rule"]["fp"] == 0
    assert metrics["ml"]["fn"] == 1
    ambiguous = scenarios.set_index("scenario").loc["uncertain"]
    assert ambiguous["evaluation_scope"] == "ambiguous_review"
    assert ambiguous["rule_detected"]
    assert not ambiguous["ml_detected"]


def test_stress_evaluation_returns_finite_model_scores_and_rolling_diagnostics():
    history = make_history()
    history["true_label"] = "normal"

    summary, rows = evaluate_stress_test(history)

    assert "hybrid" in summary
    assert rows["hybrid_prediction"].eq(
        rows[["ml_prediction", "rule_prediction"]].apply(
            lambda predictions: "anomaly" if "anomaly" in predictions.values else "normal", axis=1
        )
    ).all()

    required_diagnostics = {
        "failed_logins_45m",
        "successful_logins_5m",
        "distinct_failed_users_per_ip_15m",
        "login_frequency",
        "is_new_ip",
        "is_new_device",
        "is_unusual_location",
        "location_change",
        "ml_anomaly_score",
    }
    assert required_diagnostics.issubset(rows.columns)
    assert rows["ml_anomaly_score"].notna().all()
    assert rows["failed_logins_45m"].notna().all()
    assert rows["successful_logins_5m"].notna().all()

def test_hard_negative_benign_scenarios_cross_rule_thresholds():
    stress = build_stress_scenarios(make_history())
    typos = stress[stress["scenario"] == "benign_password_typos_then_success"]
    rapid = stress[stress["scenario"] == "benign_rapid_successes_familiar_context"]

    assert len(typos) == 6
    assert typos["true_label"].eq("normal").all()
    assert typos["login_status"].tolist() == ["failed"] * 5 + ["success"]
    assert typos["failed_attempts"].max() == 5

    assert len(rapid) == 5
    assert rapid["true_label"].eq("normal").all()
    assert rapid["login_status"].eq("success").all()
    assert rapid["timestamp"].diff().dropna().dt.total_seconds().eq(45).all()


def test_error_audit_lists_false_positives_and_false_negatives():
    rows = pd.DataFrame([
        {"scenario": "benign_case", "true_label": "normal",
         "rule_prediction": "anomaly", "ml_prediction": "normal",
         "hybrid_prediction": "anomaly", "timestamp": pd.Timestamp("2026-01-01"),
         "username": "a", "ip_address": "192.0.2.1", "login_status": "failed",
         "failed_logins_45m": 5, "successful_logins_5m": 0,
         "risk_score": 3, "ml_anomaly_score": 0.1},
        {"scenario": "attack_case", "true_label": "anomaly",
         "rule_prediction": "normal", "ml_prediction": "anomaly",
         "hybrid_prediction": "anomaly", "timestamp": pd.Timestamp("2026-01-02"),
         "username": "b", "ip_address": "192.0.2.2", "login_status": "success",
         "failed_logins_45m": 0, "successful_logins_5m": 1,
         "risk_score": 0, "ml_anomaly_score": -0.1},
    ])

    audit = build_error_audit(rows)
    # The ML predictions are correct for both rows, so the error-only audit
    # correctly has no ML rows. It should still retain errors from rules/hybrid.
    assert set(audit["detector"]) == {"rules", "hybrid"}
    assert ((audit["detector"] == "rules") & (audit["error_type"] == "false_positive")).sum() == 1
    assert ((audit["detector"] == "rules") & (audit["error_type"] == "false_negative")).sum() == 1

