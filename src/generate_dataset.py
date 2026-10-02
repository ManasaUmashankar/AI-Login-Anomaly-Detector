import random
from datetime import datetime, timedelta

import pandas as pd

from src.utils import DATA_PATH

random.seed(42)  # makes the "random" data the same every time

START_DATE = datetime(2026, 9, 1)

# Each user's normal habits
USERS = {
    "alice":   {"ip": "203.0.113.11", "location": "Mumbai, India",    "device": "Windows Laptop", "start": 9,  "end": 18},
    "bob":     {"ip": "203.0.113.12", "location": "Pune, India",      "device": "MacBook",        "start": 8,  "end": 17},
    "charlie": {"ip": "203.0.113.13", "location": "Delhi, India",     "device": "Android Phone",  "start": 10, "end": 20},
    "diana":   {"ip": "203.0.113.14", "location": "Bengaluru, India", "device": "Windows Laptop", "start": 9,  "end": 17},
}

# The browser name that goes with each device
USER_AGENTS = {
    "Windows Laptop": "Chrome on Windows",
    "MacBook": "Safari on Mac",
    "Android Phone": "Chrome on Android",
    "Unknown Device": "python-requests",
}

# Fake "attacker" IPs and places (reserved test addresses, not real people)
BAD_IPS = ["198.51.100.23", "198.51.100.77", "192.0.2.45"]
BAD_PLACES = ["Moscow, Russia", "Lagos, Nigeria", "Sao Paulo, Brazil"]


def random_time(start_hour, end_hour):
    """Pick a random moment on a random day between two hours."""
    day = START_DATE + timedelta(days=random.randrange(30))
    return day + timedelta(
        hours=random.randint(start_hour, end_hour),
        minutes=random.randrange(60),
        seconds=random.randrange(60),
    )


def make_row(time, username, ip, location, device, status, failed_attempts):
    """Build ONE log line (one row of the CSV) as a dictionary."""
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
    }


def make_normal_rows(count):
    """Normal logins: usual IP, device, hours. Sometimes a small typo."""
    rows = []
    for _ in range(count):
        name = random.choice(list(USERS))
        user = USERS[name]
        time = random_time(user["start"], user["end"])
        if random.random() < 0.05:  # 5% of the time: one typo
            status, failed = "failed", 1
        else:
            status, failed = "success", 0
        rows.append(make_row(time, name, user["ip"], user["location"], user["device"], status, failed))
    return rows


def make_brute_force_rows(name, ends_in_success):
    """Many failures within seconds from a strange IP."""
    time = random_time(0, 23)
    ip = random.choice(BAD_IPS)
    place = random.choice(BAD_PLACES)
    tries = random.randint(6, 12)
    rows = []
    for number in range(1, tries + 1):
        rows.append(make_row(time, name, ip, place, "Unknown Device", "failed", number))
        time += timedelta(seconds=random.randint(1, 4))
    if ends_in_success:
        rows.append(make_row(time, name, ip, place, "Unknown Device", "success", tries))
    return rows


def make_odd_login_row(name):
    """One successful login: new IP, new device, far country, 1-4 AM."""
    time = random_time(1, 4)
    return make_row(time, name, random.choice(BAD_IPS), random.choice(BAD_PLACES),
                    "Unknown Device", "success", 0)


def make_rapid_rows(name):
    """6 logins, 10 seconds apart, from a strange IP."""
    time = random_time(0, 23)
    ip = random.choice(BAD_IPS)
    place = random.choice(BAD_PLACES)
    rows = []
    for _ in range(6):
        rows.append(make_row(time, name, ip, place, "Unknown Device", "success", 0))
        time += timedelta(seconds=10)
    return rows


def main():
    names = list(USERS)
    rows = make_normal_rows(1500)

    for i in range(5):                       # 5 brute-force events
        rows += make_brute_force_rows(random.choice(names), ends_in_success=(i < 3))
    for _ in range(8):                       # 8 odd logins
        rows.append(make_odd_login_row(random.choice(names)))
    for _ in range(4):                       # 4 rapid-login bursts
        rows += make_rapid_rows(random.choice(names))

    df = pd.DataFrame(rows)                  # list of dictionaries -> table
    df = df.sort_values("timestamp").reset_index(drop=True)

    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} records to {DATA_PATH}")


if __name__ == "__main__":
    main()