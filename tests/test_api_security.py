import pytest

from src.api_security import SlidingWindowRateLimiter, resolve_client_identifier


def test_sliding_window_limiter_allows_limit_then_blocks():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)
    assert limiter.allow("client-a", now=100) == (True, 0)
    assert limiter.allow("client-a", now=101) == (True, 0)
    allowed, retry = limiter.allow("client-a", now=102)
    assert not allowed
    assert retry == 8


def test_sliding_window_expires_old_requests():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=10)
    assert limiter.allow("client-a", now=100)[0]
    assert limiter.allow("client-a", now=111) == (True, 0)


def test_rate_limits_are_isolated_by_client():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=10)
    assert limiter.allow("client-a", now=100)[0]
    assert limiter.allow("client-b", now=100)[0]


def test_untrusted_peer_cannot_spoof_forwarded_for():
    resolved = resolve_client_identifier(
        "203.0.113.10", "198.51.100.20", trusted_proxy_cidrs=("10.0.0.0/8",)
    )
    assert resolved == "203.0.113.10"


def test_trusted_proxy_uses_rightmost_forwarded_client_ip():
    resolved = resolve_client_identifier(
        "10.0.0.5",
        "192.0.2.10, 198.51.100.20",
        trusted_proxy_cidrs=("10.0.0.0/8",),
    )
    assert resolved == "198.51.100.20"


def test_trusted_proxy_malformed_header_falls_back_to_peer():
    resolved = resolve_client_identifier(
        "10.0.0.5", "not-an-ip", trusted_proxy_cidrs=("10.0.0.0/8",)
    )
    assert resolved == "10.0.0.5"


def test_invalid_trusted_proxy_cidr_does_not_grant_trust():
    resolved = resolve_client_identifier(
        "10.0.0.5", "198.51.100.20", trusted_proxy_cidrs=("not-a-network",)
    )
    assert resolved == "10.0.0.5"
