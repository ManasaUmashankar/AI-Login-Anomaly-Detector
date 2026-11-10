from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_invalid_login_status():
    response = client.post("/predict", json={
        "timestamp": "2026-11-10T10:00:00",
        "username": "alice",
        "ip_address": "203.0.113.10",
        "location": "Bengaluru, India",
        "device": "Windows Laptop",
        "login_status": "maybe",
        "failed_attempts": 0,
    })
    assert response.status_code == 422


def test_predict_rejects_negative_failure_count():
    response = client.post("/predict", json={
        "timestamp": "2026-11-10T10:00:00",
        "username": "alice",
        "ip_address": "203.0.113.10",
        "location": "Bengaluru, India",
        "device": "Windows Laptop",
        "login_status": "failed",
        "failed_attempts": -1,
    })
    assert response.status_code == 422


def test_predict_returns_explanation(monkeypatch):
    import pandas as pd

    history = pd.DataFrame([{
        "timestamp": "2026-11-09 10:00:00",
        "username": "alice",
        "ip_address": "203.0.113.10",
        "location": "Bengaluru, India",
        "device": "Windows Laptop",
        "login_status": "success",
        "failed_attempts": 0,
        "user_agent": "Chrome",
        "session_id": "history-1",
    }])

    monkeypatch.setattr(api, "load_model", lambda: object())
    monkeypatch.setattr(api, "load_logs", lambda: history)

    def fake_preprocess(frame, verbose=False):
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame.sort_values("timestamp").reset_index(drop=True)

    def fake_features(frame):
        frame = frame.copy()
        frame["login_hour"] = 10
        frame["failed_attempts"] = frame["failed_attempts"].astype(int)
        for name in (
            "is_new_ip", "is_new_device", "is_unusual_hour",
            "is_unusual_location", "location_change",
            "login_frequency", "failed_logins_45m", "successful_logins_5m",
            "distinct_failed_users_per_ip_15m",
        ):
            frame[name] = 0
        return frame

    def fake_apply_model(model, frame):
        frame = frame.copy()
        frame["ml_result"] = "Normal"
        frame["anomaly_score"] = 0.25
        return frame

    monkeypatch.setattr(api, "preprocess_logs", fake_preprocess)
    monkeypatch.setattr(api, "build_features", fake_features)
    monkeypatch.setattr(api, "apply_model", fake_apply_model)

    response = client.post("/predict", json={
        "timestamp": "2026-11-10T10:00:00",
        "username": "alice",
        "ip_address": "203.0.113.10",
        "location": "Bengaluru, India",
        "device": "Windows Laptop",
        "login_status": "success",
        "failed_attempts": 0,
    })

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "Low"
    assert body["ml_result"] == "Normal"
    assert body["final_status"] == "Normal activity"
    assert isinstance(body["detection_reasons"], list)
    assert "not proof" in body["disclaimer"]
