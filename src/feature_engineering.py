import pandas as pd
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

UNUSUAL_BEFORE_HOUR = 6
UNUSUAL_FROM_HOUR = 22
FREQUENCY_WINDOW_MINUTES = 5
FAILED_LOGIN_WINDOW_MINUTES = 45
SLOW_BRUTE_FORCE_WINDOW_HOURS = 6
SUCCESS_LOGIN_WINDOW_MINUTES = 5
FAILED_SOURCE_USER_WINDOW_MINUTES = 15
MINIMUM_SUCCESS_HISTORY = 3

FLAG_COLUMNS = ["is_unusual_hour", "is_new_ip", "is_new_device", "is_unusual_location", "location_change", "impossible_travel"]
NEW_COLUMNS = ["login_hour", "day_of_week"] + FLAG_COLUMNS + ["login_frequency", "failed_logins_45m", "failed_logins_6h", "successful_logins_5m", "distinct_failed_users_per_ip_15m", "travel_speed_kmh"]

def add_time_features(df):
    df["login_hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_unusual_hour"] = ((df["login_hour"] < UNUSUAL_BEFORE_HOUR) | (df["login_hour"] >= UNUSUAL_FROM_HOUR)).astype(int)
    return df

def add_difference_from_usual(df, column, new_name, min_history=1):
    """Compare to earlier successful events only; never learn from failed or incoming events."""
    flags = pd.Series(0, index=df.index, dtype="int64")
    ordered = df.sort_values("timestamp", kind="stable")
    for _, group in ordered.groupby("username", sort=False):
        history = {}
        for _, batch in group.groupby("timestamp", sort=True):
            if sum(history.values()) >= min_history:
                usual = min(history, key=lambda value: (-history[value], str(value)))
                flags.loc[batch.index] = (batch[column] != usual).astype(int)
            trusted = batch[batch["login_status"] == "success"]
            if "_incoming_event" in trusted.columns:
                trusted = trusted[~trusted["_incoming_event"].fillna(False)]
            for value in trusted[column].tolist(): history[value] = history.get(value, 0) + 1
    df[new_name] = flags
    return df

def add_location_change(df):
    flags = pd.Series(0, index=df.index, dtype="int64")
    ordered = df.sort_values(["username", "timestamp"], kind="stable")
    for _, group in ordered.groupby("username", sort=False):
        previous = None
        for _, batch in group.groupby("timestamp", sort=True):
            if previous is not None: flags.loc[batch.index] = (batch["location"] != previous).astype(int)
            successful = batch[batch["login_status"] == "success"]
            if "_incoming_event" in successful.columns: successful = successful[~successful["_incoming_event"].fillna(False)]
            if not successful.empty: previous = successful["location"].mode().sort_values().iloc[0]
    df["location_change"] = flags
    return df

def _rolling_counts(df, minutes=None, hours=None, status="failed", by_user=True):
    window = pd.Timedelta(minutes=minutes) if minutes is not None else pd.Timedelta(hours=hours)
    counts = pd.Series(0, index=df.index, dtype="int64")
    keys = ["username"] if by_user else ["ip_address"]
    ordered = df.sort_values(keys + ["timestamp"], kind="stable")
    for _, group in ordered.groupby(keys, sort=False):
        prior = []
        for _, batch in group.groupby("timestamp", sort=True):
            now = batch["timestamp"].iloc[0]
            prior = [ts for ts in prior if now - ts <= window]
            current = batch.loc[batch["login_status"] == status, "timestamp"].tolist()
            counts.loc[batch.index] = len(prior) + len(current)
            prior.extend(current)
    return counts

def add_login_frequency(df):
    window = pd.Timedelta(minutes=FREQUENCY_WINDOW_MINUTES)
    counts = pd.Series(0, index=df.index, dtype="int64")
    for _, group in df.sort_values(["username", "timestamp"], kind="stable").groupby("username", sort=False):
        prior = []
        for _, batch in group.groupby("timestamp", sort=True):
            now = batch["timestamp"].iloc[0]; prior = [ts for ts in prior if now - ts <= window]
            counts.loc[batch.index] = len(prior) + len(batch)
            prior.extend(batch["timestamp"].tolist())
    df["login_frequency"] = counts
    return df

def add_login_status_windows(df):
    df["failed_logins_45m"] = _rolling_counts(df, minutes=FAILED_LOGIN_WINDOW_MINUTES, status="failed")
    df["failed_logins_6h"] = _rolling_counts(df, hours=SLOW_BRUTE_FORCE_WINDOW_HOURS, status="failed")
    df["successful_logins_5m"] = _rolling_counts(df, minutes=SUCCESS_LOGIN_WINDOW_MINUTES, status="success")
    return df

def add_failed_users_per_ip_window(df):
    counts = pd.Series(0, index=df.index, dtype="int64")
    window = pd.Timedelta(minutes=FAILED_SOURCE_USER_WINDOW_MINUTES)
    for _, group in df.sort_values(["ip_address", "timestamp"], kind="stable").groupby("ip_address", sort=False):
        history = []
        for _, batch in group.groupby("timestamp", sort=True):
            now = batch["timestamp"].iloc[0]; history = [(t, u) for t, u in history if now - t <= window]
            current = set(batch.loc[batch["login_status"] == "failed", "username"].tolist())
            users = {u for _, u in history} | current
            counts.loc[batch.index] = len(users)
            history.extend((now, u) for u in current)
    df["distinct_failed_users_per_ip_15m"] = counts
    return df

def _haversine_km(lat1, lon1, lat2, lon2):
    from math import asin, cos, radians, sin, sqrt
    radius = 6371.0; dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * radius * asin(sqrt(min(1.0, max(0.0, a))))

def add_impossible_travel(df, threshold_kmh=900.0):
    """Flag implausible travel between trusted, coordinate-bearing successes."""
    df = df.copy()
    for column in ("latitude", "longitude"):
        if column not in df.columns:
            df[column] = float("nan")
        df[column] = pd.to_numeric(df[column], errors="coerce")
    flags = pd.Series(0, index=df.index, dtype="int64"); speeds = pd.Series(0.0, index=df.index, dtype="float64")
    for _, group in df.sort_values(["username", "timestamp"], kind="stable").groupby("username", sort=False):
        previous = None
        for _, batch in group.groupby("timestamp", sort=True):
            for idx, row in batch[batch["login_status"] == "success"].iterrows():
                is_incoming = bool(row.get("_incoming_event", False))
                if previous is not None:
                    elapsed = (row["timestamp"] - previous["timestamp"]).total_seconds() / 3600
                    coords = [previous.get("latitude"), previous.get("longitude"), row.get("latitude"), row.get("longitude")]
                    if elapsed > 0 and all(pd.notna(v) for v in coords):
                        speed = _haversine_km(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])) / elapsed
                        speeds.loc[idx] = speed; flags.loc[idx] = int(speed > threshold_kmh)
                # Incoming events are scored against history but never poison the trusted baseline.
                if not is_incoming:
                    previous = row
    df["travel_speed_kmh"] = speeds; df["impossible_travel"] = flags
    return df

def build_features(df):
    df = df.copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
    for column in ("latitude", "longitude", "scenario_id"):
        if column not in df.columns: df[column] = float("nan") if column != "scenario_id" else ""
    df = add_time_features(df)
    df = add_difference_from_usual(df, "ip_address", "is_new_ip", min_history=MINIMUM_SUCCESS_HISTORY)
    df = add_difference_from_usual(df, "device", "is_new_device", min_history=MINIMUM_SUCCESS_HISTORY)
    df = add_difference_from_usual(df, "location", "is_unusual_location", min_history=MINIMUM_SUCCESS_HISTORY)
    df = add_location_change(df)
    df = add_login_frequency(df)
    df = add_login_status_windows(df)
    df = add_failed_users_per_ip_window(df)
    return add_impossible_travel(df)

def main():
    features = build_features(preprocess_logs(load_logs()))
    print(features[NEW_COLUMNS].head())
    print(features[FLAG_COLUMNS].sum())

if __name__ == "__main__": main()
