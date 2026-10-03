from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "login_logs.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_model.pkl"