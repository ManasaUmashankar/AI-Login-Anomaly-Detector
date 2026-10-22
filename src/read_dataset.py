
import pandas as pd

from src.utils import DATA_PATH

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

REQUIRED_COLUMNS = [
    "timestamp",
    "username",
    "ip_address",
    "location",
    "device",
    "login_status",
    "user_agent",
    "session_id",
    "latitude",
    "longitude",
    "scenario_id",
]


def load_logs(include_labels=False):
    """Load login logs, optionally preserving ground-truth labels."""
    df = pd.read_csv(DATA_PATH)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if include_labels:
        if "true_label" not in df.columns:
            raise ValueError("Dataset is missing the 'true_label' column.")
        return df.copy()

    return df[REQUIRED_COLUMNS].copy()


def show_section(title):
    print(f"\n===== {title} =====")


def explore(df):
    show_section("FIRST 5 ROWS")
    print(df.head())

    show_section("LAST 5 ROWS")
    print(df.tail())

    show_section("SHAPE")
    print(df.shape)

    show_section("COLUMN NAMES")
    print(list(df.columns))

    show_section("INFO")
    df.info()

    show_section("DESCRIPTIVE STATISTICS")
    print(df.describe())

    show_section("FAILED LOGINS")
    failed = df[df["login_status"] == "failed"]
    print(f"Number of failed logins: {len(failed)}")

    show_section("ROWS WITH 5+ FAILED ATTEMPTS")
    print(
        df.loc[
            df["failed_attempts"] >= 5,
            ["timestamp", "username", "ip_address"],
        ]
    )

    show_section("LOGIN STATUS COUNTS")
    print(df["login_status"].value_counts())

    show_section("LOGINS PER USER")
    print(df["username"].value_counts())

    show_section("LOGINS PER LOCATION")
    print(df["location"].value_counts())

    show_section("LOGINS PER DEVICE")
    print(df["device"].value_counts())


def main():
    df = load_logs()
    explore(df)


if __name__ == "__main__":
    main()
