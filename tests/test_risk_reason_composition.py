"""Tests that every configured risk signal has a usable analyst explanation."""
import pandas as pd

from src.detection_rules import RISK_SIGNALS, SIGNAL_REASONS, apply_rules


def test_every_risk_signal_has_a_nonempty_explanation():
    signal_names = {name for name, _points, _function in RISK_SIGNALS}

    assert set(SIGNAL_REASONS) == signal_names
    assert all(isinstance(reason, str) and reason.strip() for reason in SIGNAL_REASONS.values())


def test_triggered_risk_signals_are_explained_without_duplicate_reasons():
    # This deliberately triggers every signal so the test can verify the complete
    # mapping in one deterministic row without needing a dataset or trained model.
    features = pd.DataFrame([{
        "failed_logins_45m": 6,
        "login_status": "success",
        "is_new_ip": 1,
        "is_unusual_hour": 1,
        "is_new_device": 1,
        "is_unusual_location": 1,
        "login_frequency": 6,
        "successful_logins_5m": 6,
        "distinct_failed_users_per_ip_15m": 4,
        "failed_logins_6h": 8,
        "impossible_travel": 1,
        "travel_speed_kmh": 1000.0,
    }])

    result = apply_rules(features)
    reasons = [reason.strip() for reason in result.loc[0, "detection_reasons"].split(";")]

    for reason in SIGNAL_REASONS.values():
        assert reason in reasons

    assert len(reasons) == len(set(reasons))
