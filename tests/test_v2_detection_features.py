import pandas as pd
from src.feature_engineering import add_impossible_travel, add_login_status_windows, add_difference_from_usual
from src.detection_rules import apply_rules, apply_risk_scores

def test_impossible_travel_flags_unrealistic_successful_login():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 09:00:00", "2026-01-01 10:00:00"]),
        "username": ["alice", "alice"], "login_status": ["success", "success"],
        "latitude": [12.9716, 51.5072], "longitude": [77.5946, -0.1276],
    })
    result = add_impossible_travel(df)
    assert result.loc[1, "impossible_travel"] == 1
    assert result.loc[1, "travel_speed_kmh"] > 900

def test_six_hour_window_catches_slow_failures():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 09:00:00", "2026-01-01 11:00:00", "2026-01-01 13:00:00"]),
        "username": ["alice"] * 3, "login_status": ["failed"] * 3,
    })
    result = add_login_status_windows(df)
    assert result["failed_logins_6h"].tolist() == [1, 2, 3]

def test_failed_login_does_not_poison_usual_ip_baseline():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 09:00:00", "2026-01-01 10:00:00", "2026-01-01 11:00:00", "2026-01-01 12:00:00", "2026-01-01 13:00:00"]),
        "username": ["alice"] * 5, "ip_address": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.9", "10.0.0.9"],
        "login_status": ["success", "success", "success", "failed", "success"],
    })
    result = add_difference_from_usual(df, "ip_address", "is_new_ip", min_history=3)
    assert result.loc[3, "is_new_ip"] == 1
    assert result.loc[4, "is_new_ip"] == 1
