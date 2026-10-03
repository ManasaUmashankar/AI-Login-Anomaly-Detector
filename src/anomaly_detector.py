import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.detection_rules import apply_rules
from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs
from src.utils import MODEL_PATH

# The columns the model is allowed to look at (all numbers)
MODEL_FEATURES = [
    "login_hour", "failed_attempts", "is_new_ip", "is_new_device",
    "is_unusual_hour", "is_unusual_location", "login_frequency", "location_change",
]

# Model settings (project choices, not universal standards)
N_TREES = 100
CONTAMINATION = 0.05     # our guess: about 5% of rows are anomalous
RANDOM_STATE = 42        # makes the result the same every run


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
    """Save the trained model to models/anomaly_model.pkl."""
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model():
    """Load the saved model (so we don't retrain every time)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.anomaly_detector")
    return joblib.load(MODEL_PATH)


def apply_model(model, features_df):
    """Add 'anomaly_score' and 'ml_result' columns."""
    df = features_df.copy()
    inputs = df[MODEL_FEATURES]

    predictions = model.predict(inputs)               # 1 = normal, -1 = anomaly
    df["anomaly_score"] = model.decision_function(inputs)   # below 0 = anomalous
    df["ml_result"] = ["Anomalous" if p == -1 else "Normal" for p in predictions]
    return df


def main():
    df = preprocess_logs(load_logs())
    features = build_features(df)

    model = train_model(features)
    save_model(model)
    print(f"\nModel trained and saved to {MODEL_PATH}")

    # Load the saved file to prove saving and loading works
    result = apply_model(load_model(), features)
    result = apply_rules(result)

    print("\nML result counts:")
    print(result["ml_result"].value_counts())

    flagged = (result["rules_triggered"] > 0).map({True: "Yes", False: "No"})
    result["flagged_by_rules"] = flagged
    print("\nML result vs rules (number of rows):")
    print(pd.crosstab(result["ml_result"], result["flagged_by_rules"]))

    print("\nMost unusual 5 rows (lowest anomaly_score):")
    lowest = result.sort_values("anomaly_score").head(5)
    print(lowest[["timestamp", "username", "ip_address", "login_status", "anomaly_score", "ml_result"]])

    missed = result[(result["ml_result"] == "Normal") & (result["rules_triggered"] > 0)]
    print(f"\nRows the rules flagged but the model called Normal: {len(missed)}")
    print(missed[["timestamp", "username", "ip_address", "login_status", "failed_attempts", "anomaly_score"]].head(5))


if __name__ == "__main__":
    main()