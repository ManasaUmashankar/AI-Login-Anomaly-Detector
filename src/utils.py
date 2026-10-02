from pathlib import Path

# Finds the main project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Where the CSV file will be saved
DATA_PATH = PROJECT_ROOT / "data" / "login_logs.csv"