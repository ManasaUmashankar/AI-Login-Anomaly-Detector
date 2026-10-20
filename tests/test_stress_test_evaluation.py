import pandas as pd

from src.preprocessing import REQUIRED_COLUMNS
from src.stress_test_evaluation import build_stress_scenarios


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

    assert len(stress) == 23
    assert set(stress["true_label"]) == {"normal", "anomaly"}
    benign = stress[stress["scenario"].str.startswith("benign_")]
    suspicious = stress[stress["scenario"].str.startswith("anomaly_")]
    assert len(benign) == 4
    assert len(suspicious) == 19
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
