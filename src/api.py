"""FastAPI inference service for the AI Login Anomaly Detector.

The endpoint scores one event using the repository's available historical logs.
This is a prototype: production deployment needs a persistent event store and
per-tenant/user history rather than a single local CSV.
"""
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

import json
import logging
from time import perf_counter

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.anomaly_detector import (
    add_ml_reason,
    apply_model,
    final_status_for,
    load_model,
)
from src.detection_rules import apply_risk_scores, apply_rules
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs
from src.mitre_mapping import map_detection_reasons
from src.api_security import enforce_rate_limit, require_api_key
from src.siem_export import build_siem_event, forward_to_configured_siems

logger = logging.getLogger("login_detector_api")

app = FastAPI(
    title="AI Login Anomaly Detector API",
    description="Explainable risk scoring and Isolation Forest anomaly detection for login events.",
    version="1.0.0",
)


@app.middleware("http")
async def log_request_metadata(request, call_next):
    """Emit structured request metadata without logging credentials or event PII."""
    started = perf_counter()
    response = await call_next(request)
    logger.info(json.dumps({
        "event": "http_request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round((perf_counter() - started) * 1000, 2),
    }))
    return response


class LoginEvent(BaseModel):
    """Validated login event accepted by POST /predict."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    username: str = Field(min_length=1, max_length=128)
    ip_address: str = Field(min_length=1, max_length=64)
    location: str = Field(min_length=1, max_length=128)
    device: str = Field(min_length=1, max_length=128)
    login_status: Literal["success", "failed"]
    failed_attempts: int = Field(ge=0, le=1000)
    user_agent: str = Field(default="unknown", min_length=1, max_length=256)
    session_id: str = Field(default_factory=lambda: str(uuid4()), max_length=128)


@app.get("/health")
def health():
    """Lightweight process health check; does not imply model readiness."""
    return {"status": "ok", "service": "ai-login-anomaly-detector"}


@app.get("/ready")
def readiness():
    """Check whether a compatible saved model can be loaded."""
    try:
        load_model()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc
    return {"status": "ready", "model": "isolation-forest"}


@app.post(
    "/predict",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
def predict(event: LoginEvent):
    """Score one event against the local historical login dataset."""
    try:
        model = load_model()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc

    history = load_logs()
    event_row = event.model_dump()
    # Generate a unique identifier to avoid colliding with historical sessions.
    event_id = f"api-{uuid4()}"
    event_row["session_id"] = event_id
    timestamp = event_row["timestamp"]
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    event_row["timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    combined = pd.concat([history, pd.DataFrame([event_row])], ignore_index=True)
    cleaned = preprocess_logs(combined, verbose=False)
    if cleaned.empty:
        raise HTTPException(status_code=422, detail="No valid login events remain after preprocessing.")

    features = build_features(cleaned)
    scored = apply_model(model, features)
    scored = apply_rules(scored)
    scored = apply_risk_scores(scored)
    scored = add_ml_reason(scored)

    matching_event = scored.loc[scored["session_id"] == event_id]
    if matching_event.empty:
        raise HTTPException(status_code=422, detail="Submitted event was removed during preprocessing.")
    result = matching_event.iloc[-1]
    final_status = final_status_for(result["risk_level"], result["ml_result"])
    reasons = [reason.strip() for reason in str(result["detection_reasons"]).split(";") if reason.strip()]
    mitre = map_detection_reasons(reasons)
    prediction = {
        "timestamp": result["timestamp"].isoformat(),
        "username": result["username"],
        "ip_address": result["ip_address"],
        "risk_score": int(result["risk_score"]),
        "risk_level": result["risk_level"],
        "ml_result": result["ml_result"],
        "anomaly_score": float(result["anomaly_score"]),
        "final_status": final_status,
        "detection_reasons": reasons,
        "mitre_attack": mitre,
    }
    siem_delivery = forward_to_configured_siems(build_siem_event(prediction))
    for destination, outcome in siem_delivery.items():
        if not outcome["sent"]:
            logger.warning(json.dumps({
                "event": "siem_delivery_failed",
                "destination": destination,
                "error_type": outcome.get("error_type", "UnknownError"),
            }))

    return {
        "timestamp": result["timestamp"].isoformat(),
        "username": result["username"],
        "risk_score": int(result["risk_score"]),
        "risk_level": result["risk_level"],
        "ml_result": result["ml_result"],
        "anomaly_score": float(result["anomaly_score"]),
        "final_status": final_status,
        "detection_reasons": reasons,
        "mitre_attack": mitre,
        "siem_delivery": siem_delivery,
        "disclaimer": "An anomaly is an investigation signal, not proof of account compromise.",
    }
