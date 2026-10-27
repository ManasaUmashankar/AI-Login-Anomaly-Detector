
import random
from datetime import datetime, timedelta

import pandas as pd

from src.utils import DATA_PATH

random.seed(42)  # Reproducible random data

START_DATE = datetime(2026, 9, 1)

# Each user's normal login habits
USERS = {
    "alice": {
        "ip": "203.0.113.11",
        "location": "Mumbai, India",
        "device": "Windows Laptop",
        "start": 9,
        "end": 18,
    },
    "bob": {
        "ip": "203.0.113.12",
        "location": "Pune, India",
        "device": "MacBook",
        "start": 8,
        "end": 17,
    },
    "charlie": {
        "ip": "203.0.113.13",
        "location": "Delhi, India",
        "device": "Android Phone",
        "start": 10,
        "end": 20,
    },
    "diana": {
        "ip": "203.0.113.14",
        "location": "Bengaluru, India",
        "device": "Windows Laptop",
        "start": 9,
        "end": 17,
    },
}

# Browser associated with each device
USER_AGENTS = {
    "Windows Laptop": "Chrome on Windows",
    "MacBook": "Safari on Mac",
    "Android Phone": "Chrome on Android",
    "Unknown Device": "python-requests",
    "iPhone": "Safari on iOS",
    "Linux Laptop": "Firefox on Linux",
}

# Reserved example IPs and fictional suspicious locations
BAD_IPS = [
    "198.51.100.23",
    "198.51.100.77",
    "192.0.2.45",
]

BAD_PLACES = [
    "Moscow, Russia",
    "Lagos, Nigeria",
    "Sao Paulo, Brazil",
]

# Benign context variation prevents one categorical field from revealing labels.
BENIGN_CONTEXTS = [
    ("198.51.100.140", "Bengaluru, India", "iPhone"),
    ("203.0.113.88", "Pune, India", "Linux Laptop"),
    ("192.0.2.140", "Mumbai, India", "Windows Laptop"),
]


def random_time(start_hour, end_hour):
    """Generate a random timestamp within the selected hours."""
    day = START_DATE + timedelta(days=random.randrange(30))

    return day + timedelta(
        hours=random.randint(start_hour, end_hour),
        minutes=random.randrange(60),
        seconds=random.randrange(60),
    )


def make_row(
    time,
    username,
    ip,
    location,
    device,
    status,
    failed_attempts,
    label="normal",
):
    """Create one login record with a ground-truth scenario label."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "ip_address": ip,
        "location": location,
        "device": device,
        "login_status": status,
        "failed_attempts": failed_attempts,
        "user_agent": USER_AGENTS[device],
        "session_id": f"{random.getrandbits(48):012x}",
        "true_label": label,
    }


def make_normal_rows(count):
    """Generate ordinary logins, including occasional failed attempts."""
    rows = []

    for _ in range(count):
        name = random.choice(list(USERS))
        user = USERS[name]

        time = random_time(user["start"], user["end"])

        # Some legitimate logins use a new device, VPN-like address, or travel context.
        # These are benign by ground truth; novelty alone is not proof of an attack.
        ip, location, device = user["ip"], user["location"], user["device"]
        if random.random() < 0.06:
            ip, location, device = random.choice(BENIGN_CONTEXTS)

        if random.random() < 0.05:
            status, failed = "failed", 1
        else:
            status, failed = "success", 0

        rows.append(
            make_row(
                time,
                name,
                ip,
                location,
                device,
                status,
                failed,
                label="normal",
            )
        )

    return rows


def make_brute_force_rows(name, ends_in_success):
    """Generate repeated failed logins, optionally followed by success."""
    time = random_time(0, 23)
    # Some attacks reuse a familiar context, so device/IP/location are not
    # individually sufficient to determine the ground-truth label.
    if random.random() < 0.25:
        user = USERS[name]
        ip, place, device = user["ip"], user["location"], user["device"]
    else:
        ip, place, device = random.choice(BAD_IPS), random.choice(BAD_PLACES), "Unknown Device"
    tries = random.randint(6, 12)

    rows = []

    for number in range(1, tries + 1):
        rows.append(
            make_row(
                time,
                name,
                ip,
                place,
                device,
                "failed",
                number,
                label="anomaly",
            )
        )

        time += timedelta(seconds=random.randint(1, 4))

    if ends_in_success:
        rows.append(
            make_row(
                time,
                name,
                ip,
                place,
                device,
                "success",
                tries,
                label="anomaly",
            )
        )

    return rows


def make_odd_login_row(name):
    """Generate a successful login at an unusual hour from a new location."""
    time = random_time(1, 4)

    if random.random() < 0.35:
        user = USERS[name]
        ip, place, device = user["ip"], user["location"], user["device"]
    else:
        ip, place, device = random.choice(BAD_IPS), random.choice(BAD_PLACES), "Unknown Device"

    return make_row(
        time,
        name,
        ip,
        place,
        device,
        "success",
        0,
        label="anomaly",
    )


def make_rapid_rows(name):
    """Generate a burst of suspicious logins ten seconds apart."""
    time = random_time(0, 23)
    if random.random() < 0.25:
        user = USERS[name]
        ip, place, device = user["ip"], user["location"], user["device"]
    else:
        ip, place, device = random.choice(BAD_IPS), random.choice(BAD_PLACES), "Unknown Device"

    rows = []

    for _ in range(6):
        rows.append(
            make_row(
                time,
                name,
                ip,
                place,
                "Unknown Device",
                "success",
                0,
                label="anomaly",
            )
        )

        time += timedelta(seconds=10)

    return rows


def main():
    names = list(USERS)

    # Generate ordinary login records
    rows = make_normal_rows(1500)

    # Generate five brute-force scenarios
    for i in range(5):
        rows += make_brute_force_rows(
            random.choice(names),
            ends_in_success=(i < 3),
        )

    # Generate eight unusual login scenarios
    for _ in range(8):
        rows.append(make_odd_login_row(random.choice(names)))

    # Generate four rapid-login bursts
    for _ in range(4):
        rows += make_rapid_rows(random.choice(names))

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save the labelled synthetic dataset
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"Saved {len(df)} records to {DATA_PATH}")

    print("\nGround-truth label counts:")
    print(df["true_label"].value_counts())

    print("\nLabel proportions:")
    print(
        (df["true_label"].value_counts(normalize=True) * 100)
        .round(2)
        .astype(str)
        .add("%")
    )


if __name__ == "__main__":
    main()
