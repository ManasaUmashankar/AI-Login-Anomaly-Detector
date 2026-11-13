import pandas as pd
import pytest

from src.openssh_ingest import load_openssh_auth_log, parse_openssh_auth_lines
from src.preprocessing import REQUIRED_COLUMNS, preprocess_logs


SAMPLE = [
    "Nov 10 09:10:00 web01 sshd[100]: Failed password for alice from 203.0.113.8 port 52210 ssh2",
    "Nov 10 09:10:05 web01 sshd[101]: Failed password for alice from 203.0.113.8 port 52211 ssh2",
    "Nov 10 09:10:10 web01 sshd[102]: Accepted password for alice from 203.0.113.8 port 52212 ssh2",
    "Nov 10 09:11:00 web01 sshd[103]: Failed password for invalid user admin from 198.51.100.9 port 60000 ssh2",
    "Nov 10 09:12:00 web01 sshd[104]: pam_unix(sshd:session): session opened for user alice",
]


def test_parser_converts_only_recognized_authentication_events():
    frame = parse_openssh_auth_lines(SAMPLE, year=2026)

    assert list(frame.columns) == REQUIRED_COLUMNS
    assert len(frame) == 4
    assert frame["login_status"].tolist() == ["failed", "failed", "success", "failed"]
    assert frame["failed_attempts"].tolist() == [1, 2, 2, 1]
    assert frame.iloc[0]["username"] == "alice"
    assert frame.iloc[3]["username"] == "admin"
    assert frame["location"].eq("unknown").all()
    assert frame["device"].eq("unknown").all()
    assert frame["session_id"].is_unique


def test_parsed_logs_are_accepted_by_existing_preprocessing():
    frame = parse_openssh_auth_lines(SAMPLE, year=2026)
    clean = preprocess_logs(frame, verbose=False)
    assert len(clean) == 4
    assert pd.api.types.is_datetime64_any_dtype(clean["timestamp"])


def test_parser_requires_valid_year():
    with pytest.raises(ValueError, match="year"):
        parse_openssh_auth_lines(SAMPLE, year=1800)


def test_file_loader_reads_auth_log(tmp_path):
    source = tmp_path / "auth.log"
    source.write_text("\n".join(SAMPLE), encoding="utf-8")
    frame = load_openssh_auth_log(source, year=2026)
    assert len(frame) == 4
