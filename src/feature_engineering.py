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
    """Compare a value with the most common value at strictly earlier timestamps.

    Events sharing a timestamp use the same baseline, so they cannot establish
    a baseline for one another. A user's first observed timestamp has no
    historical baseline; difference flags remain 0 until prior history exists.
    """
    flags = pd.Series(0, index=df.index, dtype="int64")
    ordered = df.sort_values("timestamp", kind="stable")

    for _, user_group in ordered.groupby("username", sort=False):
        history = {}
        # Process equal timestamps as a batch to avoid order-dependent leakage.
        for _, batch in user_group.groupby("timestamp", sort=True):
            if history:
                # dict insertion order gives deterministic first-seen tie-breaking.
                usual_value = max(history, key=history.get)
                flags.loc[batch.index] = (
                    batch[column] != usual_value
                ).astype("int64")

            # Only make this timestamp available to later events.
            for value in batch[column].tolist():
                history[value] = history.get(value, 0) + 1

    df[new_name] = flags
    return df

def add_location_change(df):
    """Flag location changes against the user's most recent earlier timestamp.

    Rows sharing a timestamp are evaluated as one batch. They all compare with
    the previous timestamp's location, then the batch establishes history for
    later events. This avoids row-order-dependent features for simultaneous logs.
    """
    flags = pd.Series(0, index=df.index, dtype="int64")
    ordered = df.sort_values(["username", "timestamp"], kind="stable")

    for _, user_group in ordered.groupby("username", sort=False):
        previous_location = None
        for _, batch in user_group.groupby("timestamp", sort=True):
            if previous_location is not None:
                flags.loc[batch.index] = (
                    batch["location"] != previous_location
                ).astype("int64")

            counts = batch["location"].value_counts(sort=False)
            previous_location = counts.idxmax()

    df["location_change"] = flags
    return df


def add_login_frequency(df):
    """Count trailing-window logins while batching equal timestamps.

    Simultaneous events receive the same frequency count; their row order does
    not affect the feature value.
    """
    window = pd.Timedelta(minutes=FREQUENCY_WINDOW_MINUTES)
    frequency = pd.Series(0, index=df.index, dtype="int64")
    ordered = df.sort_values(["username", "timestamp"], kind="stable")

    for _, user_group in ordered.groupby("username", sort=False):
        prior_times = []
        for _, batch in user_group.groupby("timestamp", sort=True):
            current = batch["timestamp"].iloc[0]
            prior_times = [
                timestamp for timestamp in prior_times
                if current - timestamp <= window
            ]
            frequency.loc[batch.index] = len(prior_times) + len(batch)
            prior_times.extend(batch["timestamp"].tolist())

    df["login_frequency"] = frequency
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