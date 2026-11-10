"""Parse OpenSSH sshd authentication messages into the project log schema.

Only explicit accepted/failed authentication messages are converted. The parser
does not infer geolocation or device identity; those fields remain "unknown".
"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

from src.preprocessing import REQUIRED_COLUMNS

SYSLOG_PATTERN = re.compile(
    r"^(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+"
    r"(?P<clock>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+"
    r"sshd\[(?P<pid>\d+)\]:\s+(?P<message>.*)$"
)
ACCEPTED_PATTERN = re.compile(
    r"^Accepted (?:password|publickey|keyboard-interactive/pam) for "
    r"(?P<username>\S+) from (?P<ip>\S+) port \d+\b"
)
FAILED_PATTERN = re.compile(
    r"^Failed (?:password|publickey|keyboard-interactive/pam) for "
    r"(?:invalid user )?(?P<username>\S+) from (?P<ip>\S+) port \d+\b"
)


def parse_openssh_auth_lines(lines, year=None):
    """Return recognized SSH login attempts in the canonical project schema.

    Syslog timestamps omit the year, so callers should pass the log's year when
    importing archived files. Unrecognized and non-authentication messages are
    skipped rather than guessed.
    """
    selected_year = year if year is not None else datetime.now().year
    if not isinstance(selected_year, int) or not 1970 <= selected_year <= 9999:
        raise ValueError("year must be an integer between 1970 and 9999")

    failures = defaultdict(int)
    records = []
    for line_number, raw_line in enumerate(lines, start=1):
        match = SYSLOG_PATTERN.match(raw_line.strip())
        if not match:
            continue

        message = match.group("message")
        accepted = ACCEPTED_PATTERN.match(message)
        failed = FAILED_PATTERN.match(message)
        if accepted is None and failed is None:
            continue

        parsed = accepted or failed
        username = parsed.group("username").lower()
        ip_address = parsed.group("ip")
        key = (username, ip_address)
        status = "success" if accepted else "failed"

        if status == "failed":
            failures[key] += 1
        failed_attempts = failures[key]
        if status == "success":
            failures[key] = 0

        timestamp = datetime.strptime(
            f"{match.group('month')} {match.group('day')} "
            f"{match.group('clock')} {selected_year}",
            "%b %d %H:%M:%S %Y",
        )
        records.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "username": username,
            "ip_address": ip_address,
            "location": "unknown",
            "device": "unknown",
            "login_status": status,
            "failed_attempts": failed_attempts,
            "user_agent": "OpenSSH",
            "session_id": f"openssh-{match.group('pid')}-{line_number}-{uuid4().hex[:8]}",
        })

    return pd.DataFrame(records, columns=REQUIRED_COLUMNS)


def load_openssh_auth_log(path, year=None):
    """Load an OpenSSH auth.log file using UTF-8 with replacement for bad bytes."""
    source = Path(path)
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        return parse_openssh_auth_lines(handle, year=year)
