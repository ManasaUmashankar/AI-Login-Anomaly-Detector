"""Optional Splunk HEC and Elasticsearch event forwarding.

Credentials are read only from environment variables. TLS certificate
verification stays enabled. Failures are returned to the caller for safe logging.
"""
import os

import requests


def build_siem_event(prediction):
    """Convert an API prediction into a compact ECS-inspired security event."""
    return {
        "@timestamp": prediction["timestamp"],
        "event": {
            "kind": "alert" if prediction["final_status"] != "Normal activity" else "event",
            "category": ["authentication"],
            "type": ["info"],
            "outcome": "success" if prediction["final_status"] == "Normal activity" else "unknown",
            "severity": {"Low": 2, "Medium": 5, "High": 8}.get(prediction["risk_level"], 0),
            "module": "ai_login_anomaly_detector",
            "dataset": "ai_login_anomaly_detector.authentication",
            "reason": "; ".join(prediction.get("detection_reasons", [])),
        },
        "user": {"name": prediction["username"]},
        "source": {"ip": prediction.get("ip_address", "unknown")},
        "risk": {
            "score": prediction["risk_score"],
            "level": prediction["risk_level"],
        },
        "ml": {
            "result": prediction["ml_result"],
            "anomaly_score": prediction["anomaly_score"],
        },
        "threat": {"framework": "MITRE ATT&CK", "technique": prediction.get("mitre_attack", [])},
        "labels": {"final_status": prediction["final_status"]},
    }


def forward_to_splunk(event, *, url=None, token=None, timeout=None):
    """Send one event to Splunk HTTP Event Collector (HEC)."""
    endpoint = url or os.getenv("SPLUNK_HEC_URL")
    secret = token or os.getenv("SPLUNK_HEC_TOKEN")
    if not endpoint or not secret:
        raise ValueError("SPLUNK_HEC_URL and SPLUNK_HEC_TOKEN are required.")
    response = requests.post(
        endpoint.rstrip("/") + "/services/collector/event",
        headers={"Authorization": f"Splunk {secret}"},
        json={
            "time": event["@timestamp"],
            "sourcetype": "ai_login_anomaly_detector",
            "source": "ai-login-anomaly-detector",
            "event": event,
        },
        timeout=timeout or float(os.getenv("SIEM_TIMEOUT_SECONDS", "3")),
        verify=True,
    )
    response.raise_for_status()
    return response.status_code


def forward_to_elasticsearch(event, *, url=None, api_key=None, index=None, timeout=None):
    """Index one event in Elasticsearch using an API key and verified TLS."""
    endpoint = url or os.getenv("ELASTICSEARCH_URL")
    secret = api_key or os.getenv("ELASTICSEARCH_API_KEY")
    target_index = index or os.getenv("ELASTICSEARCH_INDEX", "ai-login-anomaly-detector")
    if not endpoint or not secret:
        raise ValueError("ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY are required.")
    response = requests.post(
        endpoint.rstrip("/") + "/" + target_index + "/_doc",
        headers={"Authorization": f"ApiKey {secret}"},
        json=event,
        timeout=timeout or float(os.getenv("SIEM_TIMEOUT_SECONDS", "3")),
        verify=True,
    )
    response.raise_for_status()
    return response.status_code


def forward_to_configured_siems(event):
    """Attempt configured outputs independently; never return secrets or payloads."""
    outcomes = {}
    if os.getenv("SPLUNK_HEC_URL") or os.getenv("SPLUNK_HEC_TOKEN"):
        try:
            outcomes["splunk"] = {"sent": True, "status_code": forward_to_splunk(event)}
        except Exception as exc:
            outcomes["splunk"] = {"sent": False, "error_type": type(exc).__name__}
    if os.getenv("ELASTICSEARCH_URL") or os.getenv("ELASTICSEARCH_API_KEY"):
        try:
            outcomes["elasticsearch"] = {
                "sent": True,
                "status_code": forward_to_elasticsearch(event),
            }
        except Exception as exc:
            outcomes["elasticsearch"] = {"sent": False, "error_type": type(exc).__name__}
    return outcomes
