from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

# Project-specific thresholds (our design choices, not universal standards)
FAILED_ATTEMPTS_THRESHOLD = 5
FAILURES_BEFORE_SUCCESS_THRESHOLD = 3
RAPID_LOGIN_THRESHOLD = 5


def rule_multiple_failures(df):
    """Rule 1: many failed attempts in a row."""
    return df["failed_attempts"] >= FAILED_ATTEMPTS_THRESHOLD


def rule_success_after_failures(df):
    """Rule 2: a successful login that came after repeated failures."""
    is_success = df["login_status"] == "success"
    had_failures = df["failed_attempts"] >= FAILURES_BEFORE_SUCCESS_THRESHOLD
    return is_success & had_failures


def rule_new_ip_unusual_hour(df):
    """Rule 3: unfamiliar IP at an odd time."""
    return (df["is_new_ip"] == 1) & (df["is_unusual_hour"] == 1)


def rule_new_device_unusual_location(df):
    """Rule 4: different device from a different place."""
    return (df["is_new_device"] == 1) & (df["is_unusual_location"] == 1)


def rule_rapid_logins(df):
    """Rule 5: too many attempts in a short time."""
    return df["login_frequency"] >= RAPID_LOGIN_THRESHOLD


# Each rule = (reason text, function that checks it)
RULES = [
    ("Multiple failed login attempts", rule_multiple_failures),
    ("Successful login after repeated failures", rule_success_after_failures),
    ("New IP address at an unusual hour", rule_new_ip_unusual_hour),
    ("New device from an unusual location", rule_new_device_unusual_location),
    ("Rapid login attempts", rule_rapid_logins),
]


def apply_rules(df):
    """Run every rule. Adds 'rules_triggered' (a count) and 'detection_reasons' (text)."""
    df = df.copy()
    df["rules_triggered"] = 0
    df["detection_reasons"] = ""

    for reason, rule_function in RULES:
        hit = rule_function(df)                  # True/False for every row
        df["rules_triggered"] += hit.astype(int)
        df.loc[hit, "detection_reasons"] += reason + "; "

    df["detection_reasons"] = df["detection_reasons"].str.rstrip("; ")
    return df


def main():
    df = preprocess_logs(load_logs())
    features = build_features(df)
    result = apply_rules(features)

    print("\nHow many rows each rule caught:")
    for reason, rule_function in RULES:
        print(f"  {reason}: {int(rule_function(result).sum())}")

    flagged = result[result["rules_triggered"] > 0]
    print(f"\nRows flagged by at least one rule: {len(flagged)} of {len(result)}")

    print("\nTop 5 rows by number of rules triggered:")
    top = flagged.sort_values("rules_triggered", ascending=False).head(5)
    for _, row in top.iterrows():
        print(f"- {row['timestamp']} | {row['username']} | {row['ip_address']} | "
              f"{row['login_status']} | rules: {row['rules_triggered']}")
        print(f"    reasons: {row['detection_reasons']}")


if __name__ == "__main__":
    main()