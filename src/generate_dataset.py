"""Reproducible, mixed-behaviour synthetic authentication log generator.

The generator deliberately avoids using a unique device/IP string as an attack
label. Scenario IDs support event- and scenario-level evaluation.
"""
import argparse
import random
from datetime import datetime, timedelta

import pandas as pd

from src.utils import DATA_PATH

START_DATE = datetime(2026, 1, 1)
DAYS = 60
USER_COUNT = 36
SEED = 42
CITIES = {
    "Bengaluru, India": (12.9716, 77.5946),
    "Mumbai, India": (19.0760, 72.8777),
    "Pune, India": (18.5204, 73.8567),
    "Delhi, India": (28.6139, 77.2090),
    "Chennai, India": (13.0827, 80.2707),
    "Hyderabad, India": (17.3850, 78.4867),
    "Kolkata, India": (22.5726, 88.3639),
    "Singapore": (1.3521, 103.8198),
    "London, UK": (51.5072, -0.1276),
}
DEVICES = ["Windows Laptop", "MacBook", "Android Phone", "iPhone", "Linux Laptop"]
AGENTS = {
    "Windows Laptop": "Chrome on Windows",
    "MacBook": "Safari on Mac",
    "Android Phone": "Chrome on Android",
    "iPhone": "Safari on iOS",
    "Linux Laptop": "Firefox on Linux",
}
IPS = [f"203.0.113.{i}" for i in range(10, 240)]


def make_row(ts, username, ip, city, device, status, label="normal", scenario_id=""):
    """Build a single event; no generator-supplied failure counter is included."""
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "ip_address": ip,
        "location": city,
        "latitude": CITIES[city][0],
        "longitude": CITIES[city][1],
        "device": device,
        "login_status": status,
        "user_agent": AGENTS[device],
        "session_id": f"{random.getrandbits(48):012x}",
        "true_label": label,
        "scenario_id": scenario_id,
    }


def main(seed=SEED, output_path=DATA_PATH):
    random.seed(seed)
    users = {}
    for i in range(USER_COUNT):
        city = list(CITIES)[i % 7]
        users[f"user_{i + 1:02d}"] = {
            "city": city,
            "ip": IPS[i],
            "device": DEVICES[i % len(DEVICES)],
            "start": random.choice([6, 7, 8, 9, 10]),
            "end": random.choice([16, 17, 18, 20, 22]),
        }

    rows = []
    # Everyday logins span 60 days. Benign edge cases are intentionally common.
    for day in range(DAYS):
        date = START_DATE + timedelta(days=day)
        for username, profile in users.items():
            if random.random() > 0.58:
                continue
            ts = date + timedelta(
                hours=random.randint(profile["start"], profile["end"]),
                minutes=random.randrange(60),
                seconds=random.randrange(60),
            )
            city, ip, device = profile["city"], profile["ip"], profile["device"]
            if random.random() < 0.12:  # ordinary VPN / rotating egress
                ip = random.choice(IPS)
            if random.random() < 0.07:  # new device can be benign
                device = random.choice(DEVICES)
            if random.random() < 0.06:  # legitimate travel
                city = random.choice(list(CITIES))
                ip = random.choice(IPS)
            if random.random() < 0.10:  # late work
                ts = ts.replace(hour=random.choice([0, 1, 2, 23]))
            status = "failed" if random.random() < 0.07 else "success"
            rows.append(make_row(ts, username, ip, city, device, status))

    # Typo bursts: benign failures, often followed by a successful login.
    for index in range(18):
        username = random.choice(list(users))
        profile = users[username]
        ts = START_DATE + timedelta(days=random.randrange(DAYS), hours=random.randint(8, 21))
        scenario = f"benign_typo_{index + 1:02d}"
        for attempt in range(random.randint(2, 4)):
            rows.append(make_row(ts, username, profile["ip"], profile["city"],
                                 profile["device"], "failed", "normal", scenario))
            ts += timedelta(seconds=random.randint(8, 35))
        if random.random() < 0.8:
            rows.append(make_row(ts, username, profile["ip"], profile["city"],
                                 profile["device"], "success", "normal", scenario))

    # Attack scenarios deliberately mix familiar and unfamiliar context.
    scenario_count = 0
    for kind in ["brute_force", "password_spray", "account_takeover",
                 "rapid_success", "low_and_slow", "impossible_travel"]:
        for _ in range(4):
            scenario_count += 1
            scenario = f"{kind}_{scenario_count:02d}"
            username = random.choice(list(users))
            profile = users[username]
            ts = START_DATE + timedelta(days=random.randrange(35, DAYS), hours=random.randint(7, 22))
            familiar = random.random() < 0.5
            ip = profile["ip"] if familiar else random.choice(IPS)
            city = profile["city"] if familiar else random.choice(list(CITIES))
            device = profile["device"] if familiar or random.random() < 0.5 else random.choice(DEVICES)
            if kind == "brute_force":
                for _attempt in range(random.randint(6, 10)):
                    rows.append(make_row(ts, username, ip, city, device, "failed", "anomaly", scenario))
                    ts += timedelta(seconds=random.randint(2, 12))
            elif kind == "password_spray":
                source_ip = ip
                for target in random.sample(list(users), 5):
                    target_profile = users[target]
                    rows.append(make_row(ts, target, source_ip, target_profile["city"],
                                         target_profile["device"], "failed", "anomaly", scenario))
                    ts += timedelta(seconds=random.randint(10, 90))
            elif kind == "account_takeover":
                rows.append(make_row(ts, username, ip, city, device, "success", "anomaly", scenario))
            elif kind == "rapid_success":
                for _attempt in range(5):
                    rows.append(make_row(ts, username, ip, city, device, "success", "anomaly", scenario))
                    ts += timedelta(seconds=30)
            elif kind == "low_and_slow":
                for _attempt in range(5):
                    rows.append(make_row(ts, username, ip, city, device, "failed", "anomaly", scenario))
                    ts += timedelta(hours=random.randint(2, 5))
            else:
                far_city = random.choice([c for c in CITIES if c != profile["city"]])
                rows.append(make_row(ts, username, profile["ip"], profile["city"],
                                     profile["device"], "success", "normal", scenario))
                rows.append(make_row(ts + timedelta(minutes=20), username, random.choice(IPS),
                                     far_city, profile["device"], "success", "anomaly", scenario))

    df = pd.DataFrame(rows).sort_values(["timestamp", "username"], kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records across {USER_COUNT} users and {DAYS} days to {output_path}")
    print("Labels:", df["true_label"].value_counts().to_dict())
    print("Attack scenarios:", df.loc[df["true_label"] == "anomaly", "scenario_id"].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=str(DATA_PATH))
    args = parser.parse_args()
    main(seed=args.seed, output_path=__import__("pathlib").Path(args.output))
