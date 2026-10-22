"""Score unseen authentication events using historical context."""
import argparse
import pandas as pd

from src.anomaly_detector import (
    add_ml_reason, apply_model, apply_rules, apply_risk_scores,
    final_status_for, load_model, model_is_current, train_model,
)
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

def score_events(new_df, history_df, model):
    """Score incoming events in chronological order without future leakage."""
    if new_df.empty:
        return new_df.copy()
    history = history_df.copy()
    incoming = new_df.copy()
    history["_incoming_event"] = False
    incoming["_incoming_event"] = True
    combined = pd.concat([history, incoming], ignore_index=True, sort=False)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    combined = combined.reset_index(drop=True)
    combined = preprocess_logs(combined)
    features = build_features(combined)
    scored = apply_model(model, features)
    scored = apply_rules(scored)
    scored = apply_risk_scores(scored)
    scored = add_ml_reason(scored)
    scored["final_status"] = [final_status_for(a, b) for a, b in zip(scored["risk_level"], scored["ml_result"])]
    return scored.loc[scored["_incoming_event"].fillna(False)].drop(columns=["_incoming_event"], errors="ignore").reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Score unseen authentication events.")
    parser.add_argument("--input", required=True, help="CSV containing events to score.")
    parser.add_argument("--history", help="Optional historical CSV; defaults to the project dataset.")
    args = parser.parse_args()
    incoming = pd.read_csv(args.input)
    history = pd.read_csv(args.history) if args.history else load_logs()
    history_clean = preprocess_logs(history)
    model = load_model() if model_is_current() else train_model(build_features(history_clean))
    result = score_events(incoming, history, model)
    cols = [c for c in ["timestamp", "username", "ip_address", "risk_score", "risk_level", "ml_result", "final_status", "detection_reasons"] if c in result.columns]
    print(result[cols].to_string(index=False))

if __name__ == "__main__":
    main()
