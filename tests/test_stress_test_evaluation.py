import pandas as pd

from src.preprocessing import REQUIRED_COLUMNS
from src.stress_test_evaluation import (
    build_stress_scenarios,
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

    assert len(stress) == 30
    assert set(stress["true_label"]) == {"normal", "anomaly"}
    benign = stress[stress["scenario"].str.startswith("benign_")]
    suspicious = stress[stress["scenario"].str.startswith("anomaly_")]
    assert len(benign) == 10
    assert len(suspicious) == 20
    assert (benign["true_label"] == "normal").all()
    assert (suspicious["true_label"] == "anomaly").all()


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
