from src.api_security import SlidingWindowRateLimiter


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
