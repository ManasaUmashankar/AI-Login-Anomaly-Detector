import pandas as pd

from src.utils import DATA_PATH

# Show all columns when printing (otherwise Pandas hides some)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def load_logs():
    """Read the CSV file into a DataFrame."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("login_logs.csv not found. Run: python -m src.generate_dataset")
        raise
    return df


def show_section(title):
    """Print a heading so the output is easy to read."""
    print(f"\n===== {title} =====")


def explore(df):
    show_section("FIRST 5 ROWS (head)")
    print(df.head())

    show_section("LAST 5 ROWS (tail)")
    print(df.tail())

    show_section("SHAPE (rows, columns)")
    print(df.shape)

    show_section("COLUMN NAMES")
    print(list(df.columns))

    show_section("INFO (types and missing values)")
    df.info()

    show_section("DESCRIBE (stats for number columns)")
    print(df.describe())

    show_section("FAILED LOGINS ONLY (filtering)")
    failed = df[df["login_status"] == "failed"]
    print(f"Number of failed logins: {len(failed)}")

    show_section("ROWS WITH 5+ FAILED ATTEMPTS IN A ROW")
    many_failures = df[df["failed_attempts"] >= 5]
    print(many_failures[["timestamp", "username", "ip_address", "failed_attempts"]])

    show_section("TOP 5 HIGHEST failed_attempts (sorting)")
    top = df.sort_values("failed_attempts", ascending=False).head(5)
    print(top[["timestamp", "username", "ip_address", "failed_attempts", "login_status"]])

    show_section("LOGIN STATUS COUNTS (value_counts)")
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