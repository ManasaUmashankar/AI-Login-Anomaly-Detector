import sys

import pandas as pd

from src.read_dataset import load_logs

REQUIRED_COLUMNS = [
    "timestamp", "username", "ip_address", "location", "device",
    "login_status", "failed_attempts", "user_agent", "session_id",
]
TEXT_COLUMNS = [
    "username", "ip_address", "location", "device",
    "login_status", "user_agent", "session_id",
]
VALID_STATUSES = ["success", "failed"]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def check_required_columns(df):
    """Stop early if the file doesn't have the columns we expect."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def clean_text_columns(df):
    """Fill gaps, trim spaces, make username/status lowercase."""
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("unknown").astype(str).str.strip()
    df["username"] = df["username"].str.lower()
    df["login_status"] = df["login_status"].str.lower()
    return df


def fix_failed_attempts(df):
    """Make failed_attempts a whole number. Unreadable or missing -> 0."""
    numbers = pd.to_numeric(df["failed_attempts"], errors="coerce")
    df["failed_attempts"] = numbers.fillna(0).astype(int)
    return df


def fix_timestamps(df):
    """Turn text into real dates. Rows with unreadable dates are removed."""
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    bad_count = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])
    return df, bad_count


def remove_duplicates(df):
    """Remove rows that are exact copies of another row."""
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)


def validate_values(df):
    """Keep only rows with a known status and a non-negative count.
    NOTE: we do NOT remove rows just because they look suspicious."""
    is_valid = df["login_status"].isin(VALID_STATUSES) & (df["failed_attempts"] >= 0)
    removed = int((~is_valid).sum())
    return df[is_valid], removed


def preprocess_logs(df):
    """Run every cleaning step and print a summary of what changed."""
    df = df.copy()
    check_required_columns(df)

    rows_at_start = len(df)
    missing_found = int(df.isna().sum().sum())

    df = clean_text_columns(df)
    df = fix_failed_attempts(df)
    df, bad_timestamps = fix_timestamps(df)
    df, duplicates = remove_duplicates(df)
    df, bad_values = validate_values(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("----- PREPROCESSING SUMMARY -----")
    print(f"Rows at start:                {rows_at_start}")
    print(f"Missing values found (filled): {missing_found}")
    print(f"Duplicate rows removed:       {duplicates}")
    print(f"Invalid timestamps removed:   {bad_timestamps}")
    print(f"Invalid values removed:       {bad_values}")
    print(f"Rows at end:                  {len(df)}")
    return df


def demo_messy_data():
    """A tiny table with deliberate problems, to prove the cleaner works."""
    rows = [
        ["2026-09-01 09:00:00", "Alice ", "203.0.113.11", "Mumbai, India", "Windows Laptop", "SUCCESS", 0, "Chrome on Windows", "s1"],
        ["2026-09-01 09:00:00", "Alice ", "203.0.113.11", "Mumbai, India", "Windows Laptop", "SUCCESS", 0, "Chrome on Windows", "s1"],
        ["not-a-date", "bob", "203.0.113.12", "Pune, India", "MacBook", "success", 0, "Safari on Mac", "s2"],
        ["2026-09-01 10:00:00", "charlie", "203.0.113.13", None, "Android Phone", "success", 0, "Chrome on Android", "s3"],
        ["2026-09-01 11:00:00", "diana", "203.0.113.14", "Bengaluru, India", "Windows Laptop", "success", None, "Chrome on Windows", "s4"],
        ["2026-09-01 12:00:00", "bob", "203.0.113.12", "Pune, India", "MacBook", "maybe", 0, "Safari on Mac", "s5"],
        ["2026-09-01 03:00:00", "alice", "198.51.100.23", "Moscow, Russia", "Unknown Device", "failed", 12, "python-requests", "s6"],
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("Running on the small MESSY demo table\n")
        df = demo_messy_data()
        show_table = True
    else:
        df = load_logs()
        show_table = False

    clean = preprocess_logs(df)
    print("\nColumn types after cleaning:")
    print(clean.dtypes)
    if show_table:
        print("\nCleaned table:")
        print(clean[["timestamp", "username", "location", "login_status", "failed_attempts"]])


if __name__ == "__main__":
    main()