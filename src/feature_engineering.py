import pandas as pd

from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

# Project-specific choices (design decisions, not universal standards)
UNUSUAL_BEFORE_HOUR = 6        # logins before 06:00 count as unusual
UNUSUAL_FROM_HOUR = 22         # logins from 22:00 onward count as unusual
FREQUENCY_WINDOW_MINUTES = 5   # window for counting recent logins

FLAG_COLUMNS = [
    "is_unusual_hour", "is_new_ip", "is_new_device",
    "is_unusual_location", "location_change",
]
NEW_COLUMNS = ["login_hour", "day_of_week"] + FLAG_COLUMNS + ["login_frequency"]


def add_time_features(df):
    """Hour, weekday, and an 'unusual hour' flag."""
    df["login_hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0 = Monday
    too_early = df["login_hour"] < UNUSUAL_BEFORE_HOUR
    too_late = df["login_hour"] >= UNUSUAL_FROM_HOUR
    df["is_unusual_hour"] = (too_early | too_late).astype(int)
    return df


def add_difference_from_usual(df, column, new_name):
    """1 if this row's value differs from the user's most common value."""
    usual = df.groupby("username")[column].agg(lambda values: values.mode().iloc[0])
    usual_for_each_row = df["username"].map(usual)
    df[new_name] = (df[column] != usual_for_each_row).astype(int)
    return df


def add_location_change(df):
    """1 if location differs from the same user's previous login."""
    previous = df.groupby("username")["location"].shift(1)
    df["location_change"] = ((df["location"] != previous) & previous.notna()).astype(int)
    return df


def add_login_frequency(df):
    """Count each user's logins in the last few minutes (sliding window)."""
    window = pd.Timedelta(minutes=FREQUENCY_WINDOW_MINUTES)
    frequency = pd.Series(0, index=df.index)

    for username, group in df.groupby("username"):
        times = list(group["timestamp"])       # already in time order
        counts = []
        start = 0
        for i, current in enumerate(times):
            # move 'start' forward until it is inside the window
            while current - times[start] > window:
                start += 1
            counts.append(i - start + 1)
        frequency.loc[group.index] = counts

    df["login_frequency"] = frequency
    return df


def build_features(df):
    """Add every feature column to a cleaned DataFrame."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_time_features(df)
    df = add_difference_from_usual(df, "ip_address", "is_new_ip")
    df = add_difference_from_usual(df, "device", "is_new_device")
    df = add_difference_from_usual(df, "location", "is_unusual_location")
    df = add_location_change(df)
    df = add_login_frequency(df)
    return df


def main():
    df = preprocess_logs(load_logs())
    features = build_features(df)

    print("\nNew columns (first 5 rows):")
    print(features[NEW_COLUMNS].head())

    print("\nHow many rows have each flag = 1:")
    print(features[FLAG_COLUMNS].sum())

    print("\nHighest login_frequency values:")
    top = features.sort_values("login_frequency", ascending=False)
    print(top[["timestamp", "username", "ip_address", "login_frequency"]].head(5))


if __name__ == "__main__":
    main()