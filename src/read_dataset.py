import pandas as pd
from src.utils import DATA_PATH
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
REQUIRED_COLUMNS = ["timestamp", "username", "ip_address", "location", "device", "login_status", "user_agent", "session_id"]
OPTIONAL_METADATA = ["latitude", "longitude", "scenario_id"]

def load_logs(include_labels=False):
    """Load authentication logs and optionally preserve synthetic evaluation labels."""
    df = pd.read_csv(DATA_PATH)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    for col in OPTIONAL_METADATA:
        if col not in df.columns: df[col] = float("nan") if col != "scenario_id" else ""
    if include_labels and "true_label" not in df.columns:
        raise ValueError("Dataset is missing the true_label column.")
    columns = REQUIRED_COLUMNS + OPTIONAL_METADATA
    if include_labels: columns += ["true_label"]
    return df[columns].copy()

def show_section(title): print(f"\n===== {title} =====")

def explore(df):
    for title, value in [("FIRST 5 ROWS", df.head()), ("LAST 5 ROWS", df.tail()), ("SHAPE", df.shape), ("COLUMN NAMES", list(df.columns))]:
        show_section(title); print(value)
    show_section("INFO"); df.info()
    show_section("DESCRIPTIVE STATISTICS"); print(df.describe(include="all"))
    show_section("FAILED LOGINS"); print(df.loc[df["login_status"] == "failed", ["timestamp", "username", "ip_address"]].head(20))
    show_section("LOGIN STATUS COUNTS"); print(df["login_status"].value_counts())
    show_section("LOGINS PER USER"); print(df["username"].value_counts())
    show_section("LOGINS PER LOCATION"); print(df["location"].value_counts())
    show_section("LOGINS PER DEVICE"); print(df["device"].value_counts())

def main(): explore(load_logs())
if __name__ == "__main__": main()
