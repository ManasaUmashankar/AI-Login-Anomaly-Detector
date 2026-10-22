import pandas as pd
from src.score_events import score_events

class AlwaysNormalModel:
    def predict(self, X): return [1] * len(X)
    def decision_function(self, X): return [0.25] * len(X)

def _event(ts, user, ip, status, sid):
    return {"timestamp": ts, "username": user, "ip_address": ip, "location": "Bengaluru, India", "device": "Windows Laptop", "login_status": status, "user_agent": "Chrome on Windows", "session_id": sid, "latitude": 12.9716, "longitude": 77.5946, "scenario_id": ""}

def test_new_user_slow_brute_force_is_scored_and_explained():
    history = pd.DataFrame([_event("2026-01-01 09:00:00", "known", "203.0.113.10", "success", "h1")])
    incoming = pd.DataFrame([_event(f"2026-01-02 {hour:02d}:{minute:02d}:00", "new_user", "198.51.100.77", "failed", f"a{i}") for i, (hour, minute) in enumerate([(0,0),(0,40),(1,20),(2,0),(2,40),(3,20),(4,0),(4,40)])])
    result = score_events(incoming, history, AlwaysNormalModel())
    assert len(result) == len(incoming)
    assert result["risk_score"].max() >= 3
    assert result["detection_reasons"].str.contains("six hours", case=False).any()
