from src.feature_engineering import build_features
from src.preprocessing import preprocess_logs
from src.read_dataset import load_logs

# Project-specific thresholds (our design choices, not universal standards)
FAILED_ATTEMPTS_THRESHOLD = 5
FAILURES_BEFORE_SUCCESS_THRESHOLD = 3
RAPID_LOGIN_THRESHOLD = 5
FAILED_LOGIN_WINDOW_THRESHOLD = 5
RAPID_SUCCESS_WINDOW_THRESHOLD = 5
FAILED_USERS_PER_SOURCE_THRESHOLD = 4

# NEW (Stage 7): risk levels (also project-specific)
LOW_MAX_SCORE = 2        # 0 to 2 = Low
MEDIUM_MAX_SCORE = 5     # 3 to 5 = Medium, 6 and above = High


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


def rule_repeated_failures_in_window(df):
    """Flag five or more failed events in the trailing 45-minute window."""
    return df["failed_logins_45m"] >= FAILED_LOGIN_WINDOW_THRESHOLD


def rule_rapid_successes(df):
    """Flag five or more successful events in the trailing five-minute window."""
    return df["successful_logins_5m"] >= RAPID_SUCCESS_WINDOW_THRESHOLD


def rule_password_spray_source(df):
    """Flag a source IP with failures against four or more accounts in 15 minutes."""
    return df["distinct_failed_users_per_ip_15m"] >= FAILED_USERS_PER_SOURCE_THRESHOLD


# Each rule = (reason text, function that checks it)
RULES = [
    ("Multiple failed login attempts", rule_multiple_failures),
    ("Successful login after repeated failures", rule_success_after_failures),
    ("New IP address at an unusual hour", rule_new_ip_unusual_hour),
    ("New device from an unusual location", rule_new_device_unusual_location),
    ("Rapid login attempts", rule_rapid_logins),
    ("Repeated failed logins within 45 minutes", rule_repeated_failures_in_window),
    ("Rapid successful login burst", rule_rapid_successes),
    ("Failed logins against multiple accounts from one source", rule_password_spray_source),
]


# NEW (Stage 7): simple yes/no signals taken straight from the features
def signal_new_ip(df):
    return df["is_new_ip"] == 1


def signal_new_device(df):
    return df["is_new_device"] == 1


def signal_unusual_location(df):
    return df["is_unusual_location"] == 1


def signal_unusual_hour(df):
    return df["is_unusual_hour"] == 1


# NEW (Stage 7): each signal = (name, points, function that checks it)
RISK_SIGNALS = [
    ("Multiple failed login attempts", 2, rule_multiple_failures),
    ("Successful login after repeated failures", 2, rule_success_after_failures),
    ("New IP address", 2, signal_new_ip),
    ("New device", 2, signal_new_device),
    ("Unusual login location", 2, signal_unusual_location),
    ("Rapid login attempts", 2, rule_rapid_logins),
    ("Unusual login hour", 1, signal_unusual_hour),
    ("Repeated failures within 45 minutes", 3, rule_repeated_failures_in_window),
    ("Rapid successful logins within 5 minutes", 3, rule_rapid_successes),
    ("Multiple accounts failed from one source within 15 minutes", 3, rule_password_spray_source),
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


# NEW (Stage 7)
def risk_level_for(score):
    """Turn a number into Low, Medium or High."""
    if score <= LOW_MAX_SCORE:
        return "Low"
    if score <= MEDIUM_MAX_SCORE:
        return "Medium"
    return "High"


# NEW (Stage 7)
def apply_risk_scores(df):
    """Add 'risk_score' (points added up) and 'risk_level' (Low/Medium/High)."""
    df = df.copy()
    df["risk_score"] = 0

    for name, points, signal_function in RISK_SIGNALS:
        hit = signal_function(df)                # True/False for every row
        df["risk_score"] += hit.astype(int) * points

    df["risk_level"] = df["risk_score"].apply(risk_level_for)
    return df


def main():
    df = preprocess_logs(load_logs())
    features = build_features(df)
    result = apply_risk_scores(apply_rules(features))

    flagged = result[result["rules_triggered"] > 0]
    print(f"\nRows flagged by at least one rule: {len(flagged)} of {len(result)}")

    print("\nRisk level counts:")
    levels = result["risk_level"].value_counts()
    print(levels.reindex(["High", "Medium", "Low"], fill_value=0))

    print("\nHow many rows have each risk score:")
    print(result["risk_score"].value_counts().sort_index())

    print("\nTop 5 highest risk scores:")
    top = result.sort_values("risk_score", ascending=False).head(5)
    print(top[["timestamp", "username", "ip_address", "login_status", "risk_score", "risk_level"]])


if __name__ == "__main__":
    main()