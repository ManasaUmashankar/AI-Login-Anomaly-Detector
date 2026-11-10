import pytest

from src.siem_export import build_siem_event, forward_to_elasticsearch, forward_to_splunk


@pytest.fixture
def event():
    return {
        "timestamp": "2026-11-10T10:00:00",
        "username": "alice",
        "ip_address": "203.0.113.10",
        "risk_score": 7,
        "risk_level": "High",
        "ml_result": "Anomalous",
        "anomaly_score": -0.2,
        "final_status": "Investigate",
        "detection_reasons": ["Multiple failed login attempts"],
        "mitre_attack": [],
    }


def test_build_siem_event_has_authentication_fields(event):
    result = build_siem_event(event)
    assert result["event"]["category"] == ["authentication"]
    assert result["user"]["name"] == "alice"
    assert result["source"]["ip"] == "203.0.113.10"
    assert result["risk"]["level"] == "High"


def test_splunk_hec_uses_token_and_verified_tls(monkeypatch, event):
    import src.siem_export as module

    captured = {}

    class Response:
        status_code = 200
        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)
    assert forward_to_splunk(build_siem_event(event), url="https://splunk.example", token="secret") == 200
    assert captured["url"] == "https://splunk.example/services/collector/event"
    assert captured["headers"]["Authorization"] == "Splunk secret"
    assert captured["verify"] is True


def test_elasticsearch_uses_api_key_and_verified_tls(monkeypatch, event):
    import src.siem_export as module

    captured = {}

    class Response:
        status_code = 201
        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(module.requests, "post", fake_post)
    assert forward_to_elasticsearch(
        build_siem_event(event),
        url="https://elastic.example",
        api_key="secret",
    ) == 201
    assert captured["url"] == "https://elastic.example/ai-login-anomaly-detector/_doc"
    assert captured["headers"]["Authorization"] == "ApiKey secret"
    assert captured["verify"] is True
