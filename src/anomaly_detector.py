import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.detection_rules import apply_risk_scores, apply_rules
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs
from src.utils import MODEL_PATH

# The columns the model is allowed to look at (all numbers)
MODEL_FEATURES = [
    "login_hour", "failed_attempts", "is_new_ip", "is_new_device",
    "is_unusual_hour", "is_unusual_location", "login_frequency", "location_change",
    "failed_logins_45m", "successful_logins_5m",
]

# Model settings (project choices, not universal standards)
N_TREES = 100
CONTAMINATION = 0.05     # our guess: about 5% of rows are anomalous
RANDOM_STATE = 42        # makes the result the same every run
FEATURE_VERSION = 3      # rolling-window features added to the model inputs

# NEW (Stage 9)
ML_REASON = "Statistically unusual pattern (ML)"
FINAL_COLUMNS = [
    "timestamp", "username", "ip_address", "location", "device", "login_status",
    "risk_score", "risk_level", "ml_result", "final_status", "detection_reasons",
]


def train_model(features_df):
    """Teach the model what the data looks like (no labels used)."""
    model = IsolationForest(
        n_estimators=N_TREES,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(features_df[MODEL_FEATURES])
    return model


def save_model(model):
    """Save the trained model to the configured MODEL_PATH."""
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"feature_version": FEATURE_VERSION, "model": model}, MODEL_PATH)

def model_is_current():
    """Return whether the saved artifact matches the current feature semantics."""
    if not MODEL_PATH.exists():
        return False

    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception:
        # A missing, legacy, or unreadable artifact should be rebuilt from data.
        return False

    return (
        isinstance(artifact, dict)
        and artifact.get("feature_version") == FEATURE_VERSION
        and "model" in artifact
    )


def load_model():
    """Load a saved model only when its feature version matches this code."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.anomaly_detector")

    artifact = joblib.load(MODEL_PATH)
    if (
        not isinstance(artifact, dict)
        or artifact.get("feature_version") != FEATURE_VERSION
        or "model" not in artifact
    ):
        raise ValueError(
            "The saved model is outdated or unversioned. "
            "Run: python -m src.anomaly_detector to retrain it."
        )
    return artifact["model"]

def apply_model(model, features_df):
    """Add 'anomaly_score' and 'ml_result' columns."""
    df = features_df.copy()
    inputs = df[MODEL_FEATURES]

    predictions = model.predict(inputs)               # 1 = normal, -1 = anomaly
    df["anomaly_score"] = model.decision_function(inputs)   # below 0 = anomalous
    df["ml_result"] = ["Anomalous" if p == -1 else "Normal" for p in predictions]
    return df


# NEW (Stage 9)
def add_ml_reason(df):
    """Add the ML reason to rows the model flagged."""
    is_anomalous = df["ml_result"] == "Anomalous"
    has_reasons = df["detection_reasons"] != ""

    df.loc[is_anomalous & has_reasons, "detection_reasons"] += "; " + ML_REASON
    df.loc[is_anomalous & ~has_reasons, "detection_reasons"] = ML_REASON
    return df


# NEW (Stage 9)
def final_status_for(risk_level, ml_result):
    """Combine the risk level and ML result into one status."""
    if risk_level == "High":
        return "Requires investigation"
    if risk_level == "Medium":
        if ml_result == "Anomalous":
            return "Requires investigation"
        return "Potentially suspicious"
    # Low risk
    if ml_result == "Anomalous":
        return "Potentially suspicious"
    return "Normal activity"


# NEW (Stage 9)
def run_pipeline(retrain=False):
    """Run every stage and return the final results table."""
    df = preprocess_logs(load_logs())
    features = build_features(df)

    if retrain or not model_is_current():
        save_model(train_model(features))
    model = load_model()

    result = apply_model(model, features)
    result = apply_rules(result)
    result = apply_risk_scores(result)
    result = add_ml_reason(result)
    result["final_status"] = [
        final_status_for(level, ml)
        for level, ml in zip(result["risk_level"], result["ml_result"])
    ]
    return result


# NEW (Stage 9)
def print_rows(rows):
    """Print rows in an easy-to-read format."""
    for _, row in rows.iterrows():
        print(f"- {row['timestamp']} | {row['username']} | {row['ip_address']} | "
              f"{row['location']} | {row['device']}")
        print(f"    score {row['risk_score']} ({row['risk_level']}) | "
              f"ML: {row['ml_result']} | status: {row['final_status']}")
        print(f"    reasons: {row['detection_reasons']}")


def main():
    result = run_pipeline(retrain=True)
    print(f"\nModel trained and saved to {MODEL_PATH}")

    print("\nFinal status counts:")
    print(result["final_status"].value_counts())

    print("\nRisk level vs ML result (number of rows):")
    print(pd.crosstab(result["risk_level"], result["ml_result"]))

    print("\nTop 3 rows that require investigation:")
    urgent = result[result["final_status"] == "Requires investigation"]
    print_rows(urgent.sort_values("risk_score", ascending=False).head(3))

    print("\nRows that are only 'Potentially suspicious':")
    print_rows(result[result["final_status"] == "Potentially suspicious"])

    print("\nHigh risk by rules, but the model said Normal (2 examples):")
    missed = result[(result["risk_level"] == "High") & (result["ml_result"] == "Normal")]
    print_rows(missed.head(2))


if __name__ == "__main__":
    main()