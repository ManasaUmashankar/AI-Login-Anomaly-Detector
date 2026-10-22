import pandas as pd

from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

# Project-specific choices (design decisions, not universal standards)
UNUSUAL_BEFORE_HOUR = 6        # logins before 06:00 count as unusual
UNUSUAL_FROM_HOUR = 22         # logins from 22:00 onward count as unusual
FREQUENCY_WINDOW_MINUTES = 5   # window for counting recent logins
FAILED_LOGIN_WINDOW_MINUTES = 45  # rolling window for repeated failed-login events
SUCCESS_LOGIN_WINDOW_MINUTES = 5  # rolling window for rapid successful logins
FAILED_SOURCE_USER_WINDOW_MINUTES = 15  # distinct accounts failing from one source IP

FLAG_COLUMNS = [
    "is_unusual_hour", "is_new_ip", "is_new_device",
    "is_unusual_location", "location_change",
]
NEW_COLUMNS = ["login_hour", "day_of_week"] + FLAG_COLUMNS + [
    "login_frequency", "failed_logins_45m", "successful_logins_5m",
    "distinct_failed_users_per_ip_15m",
]


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
                # Break frequency ties by a stable value ordering, not input row order.
                usual_value = min(
                    history,
                    key=lambda value: (-history[value], str(value)),
                )
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
            highest_count = counts.max()
            # A timestamp batch can contain tied locations; choose consistently.
            tied_locations = counts[counts == highest_count].index.tolist()
            previous_location = min(tied_locations, key=lambda value: str(value))

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


def add_login_status_windows(df):
    """Count failed and successful events in inclusive rolling windows per user.

    Equal-timestamp events are evaluated as a batch, so their feature values do
    not depend on row order. Counts include the current timestamp batch.
    """
    failed_window = pd.Timedelta(minutes=FAILED_LOGIN_WINDOW_MINUTES)
    success_window = pd.Timedelta(minutes=SUCCESS_LOGIN_WINDOW_MINUTES)
    failed_counts = pd.Series(0, index=df.index, dtype="int64")
    success_counts = pd.Series(0, index=df.index, dtype="int64")
    ordered = df.sort_values(["username", "timestamp"], kind="stable")

    for _, user_group in ordered.groupby("username", sort=False):
        prior_failed = []
        prior_failed_slow = []
        prior_success = []
        for _, batch in user_group.groupby("timestamp", sort=True):
            current = batch["timestamp"].iloc[0]
            prior_failed = [
                timestamp for timestamp in prior_failed
                if current - timestamp <= failed_window
            ]
            prior_failed_slow = [
                timestamp for timestamp in prior_failed_slow
                if current - timestamp <= slow_window
            ]
            prior_success = [
                timestamp for timestamp in prior_success
                if current - timestamp <= success_window
            ]
            batch_failed = batch.loc[batch["login_status"] == "failed", "timestamp"].tolist()
            batch_success = batch.loc[batch["login_status"] == "success", "timestamp"].tolist()
            failed_counts.loc[batch.index] = len(prior_failed) + len(batch_failed)
            slow_failed_counts.loc[batch.index] = len(prior_failed_slow) + len(batch_failed)
            success_counts.loc[batch.index] = len(prior_success) + len(batch_success)
            prior_failed.extend(batch_failed)
            prior_failed_slow.extend(batch_failed)
            prior_success.extend(batch_success)

    df["failed_logins_45m"] = failed_counts
    df["failed_logins_6h"] = slow_failed_counts
    df["successful_logins_5m"] = success_counts
    return df



def add_failed_users_per_ip_window(df):
    """Count distinct usernames with failed logins from each IP in 15 minutes.

    The window is inclusive and keyed by source IP rather than account, helping
    surface password-spraying patterns across multiple accounts. Equal timestamps
    are evaluated as a batch to avoid row-order-dependent counts.
    """
    window = pd.Timedelta(minutes=FAILED_SOURCE_USER_WINDOW_MINUTES)
    counts = pd.Series(0, index=df.index, dtype="int64")
    ordered = df.sort_values(["ip_address", "timestamp"], kind="stable")

    for _, ip_group in ordered.groupby("ip_address", sort=False):
        prior_failures = []
        for _, batch in ip_group.groupby("timestamp", sort=True):
            current = batch["timestamp"].iloc[0]
            prior_failures = [
                (timestamp, username)
                for timestamp, username in prior_failures
                if current - timestamp <= window
            ]
            current_failed_users = set(
                batch.loc[batch["login_status"] == "failed", "username"].tolist()
            )
            distinct_users = {username for _, username in prior_failures}
            distinct_users.update(current_failed_users)
            counts.loc[batch.index] = len(distinct_users)

            prior_failures.extend(
                (current, username)
                for username in batch.loc[
                    batch["login_status"] == "failed", "username"
                ].tolist()
            )

    df["distinct_failed_users_per_ip_15m"] = counts
    return df



def _haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two coordinate pairs."""
    from math import asin, cos, radians, sin, sqrt
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(min(1.0, max(0.0, a))))


def add_impossible_travel(df, threshold_kmh=900.0):
    """Flag a successful login requiring implausible travel since prior success."""
    flags = pd.Series(0, index=df.index, dtype="int64")
    speeds = pd.Series(0.0, index=df.index, dtype="float64")
    ordered = df.sort_values(["username", "timestamp"], kind="stable")
    for _, group in ordered.groupby("username", sort=False):
        previous = None
        for _, batch in group.groupby("timestamp", sort=True):
            for idx, row in batch[batch["login_status"] == "success"].iterrows():
                if previous is not None:
                    elapsed = (row["timestamp"] - previous["timestamp"]).total_seconds() / 3600
                    coords = [previous.get("latitude"), previous.get("longitude"), row.get("latitude"), row.get("longitude")]
                    if elapsed > 0 and all(pd.notna(v) for v in coords):
                        distance = _haversine_km(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
                        speed = distance / elapsed
                        speeds.loc[idx] = speed
                        flags.loc[idx] = int(speed > threshold_kmh)
                previous = row
    df["travel_speed_kmh"] = speeds
    df["impossible_travel"] = flags
    return df


def build_features(df):
    """Add all model features using strictly historical, timestamp-safe baselines."""
    df = df.copy()
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = add_time_features(df)
    df = add_difference_from_usual(df, "ip_address", "is_new_ip")
    df = add_difference_from_usual(df, "device", "is_new_device")
    df = add_difference_from_usual(df, "location", "is_unusual_location")
    df = add_location_change(df)
    df = add_login_frequency(df)
    df = add_login_status_windows(df)
    df = add_failed_users_per_ip_window(df)
    df = add_impossible_travel(df)
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