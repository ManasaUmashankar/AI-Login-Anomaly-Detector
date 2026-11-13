import pandas as pd

from src.feature_engineering import build_features


def _logs(rows):
    return pd.DataFrame(rows, columns=[
        "timestamp", "username", "ip_address", "location", "device",
        "login_status", "failed_attempts", "user_agent", "session_id",
    ]).assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))


def test_future_events_do_not_change_earlier_features():
    base = _logs([
        ("2026-11-10 09:00:00", "alice", "203.0.113.10", "Bengaluru", "Laptop", "failed", 1, "Chrome", "a1"),
        ("2026-11-10 09:02:00", "alice", "203.0.113.10", "Bengaluru", "Laptop", "failed", 2, "Chrome", "a2"),
    ])
    extended = pd.concat([base, _logs([
        ("2026-11-10 09:04:00", "alice", "198.51.100.20", "Delhi", "Phone", "success", 0, "Firefox", "future-1"),
        ("2026-11-10 09:05:00", "bob", "203.0.113.10", "Pune", "Tablet", "failed", 1, "Edge", "future-2"),
    ])], ignore_index=True)

    base_features = build_features(base).set_index("session_id")
    extended_features = build_features(extended).set_index("session_id")
    columns = [
        "is_new_ip", "is_new_device", "is_unusual_location", "location_change",
        "login_frequency", "failed_logins_45m", "successful_logins_5m",
        "distinct_failed_users_per_ip_15m",
    ]

    pd.testing.assert_frame_equal(
        base_features.loc[["a1", "a2"], columns],
        extended_features.loc[["a1", "a2"], columns],
    )


def test_equal_timestamp_features_are_independent_of_input_order():
    rows = [
        ("2026-11-10 09:00:00", "alice", "203.0.113.10", "Bengaluru", "Laptop", "failed", 1, "Chrome", "prior"),
        ("2026-11-10 09:01:00", "alice", "203.0.113.11", "Delhi", "Phone", "failed", 1, "Firefox", "same-1"),
        ("2026-11-10 09:01:00", "alice", "203.0.113.12", "Pune", "Tablet", "success", 0, "Edge", "same-2"),
        ("2026-11-10 09:01:00", "bob", "203.0.113.11", "Delhi", "Phone", "failed", 1, "Firefox", "same-3"),
    ]
    forward = build_features(_logs(rows)).set_index("session_id").sort_index()
    reversed_order = build_features(_logs(list(reversed(rows)))).set_index("session_id").sort_index()

    columns = [
        "is_new_ip", "is_new_device", "is_unusual_location", "location_change",
        "login_frequency", "failed_logins_45m", "successful_logins_5m",
        "distinct_failed_users_per_ip_15m",
    ]
    pd.testing.assert_frame_equal(forward[columns], reversed_order[columns])


def test_event_older_than_latest_log_uses_only_earlier_history():
    rows = [
        ("2026-11-10 09:00:00", "alice", "203.0.113.10", "Bengaluru", "Laptop", "success", 0, "Chrome", "before"),
        ("2026-11-10 11:00:00", "alice", "198.51.100.20", "Delhi", "Phone", "failed", 1, "Firefox", "after"),
        ("2026-11-10 10:00:00", "alice", "203.0.113.10", "Bengaluru", "Laptop", "success", 0, "Chrome", "submitted"),
    ]
    features = build_features(_logs(rows)).set_index("session_id")
    submitted = features.loc["submitted"]

    assert submitted["is_new_ip"] == 0
    assert submitted["is_new_device"] == 0
    assert submitted["is_unusual_location"] == 0
    assert submitted["login_frequency"] == 1
    assert submitted["successful_logins_5m"] == 1
