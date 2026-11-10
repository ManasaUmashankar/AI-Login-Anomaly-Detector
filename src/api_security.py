"""Small, dependency-free API-key and sliding-window rate-limit helpers."""
from collections import defaultdict, deque
from hmac import compare_digest
import os
from threading import Lock
from time import monotonic

from fastapi import Header, HTTPException, Request

API_KEY = os.getenv("DETECTOR_API_KEY", "")
RATE_LIMIT_REQUESTS = int(os.getenv("DETECTOR_RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("DETECTOR_RATE_LIMIT_WINDOW_SECONDS", "60"))
_MAX_TRACKED_CLIENTS = 10000


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Fail closed if no server key is configured or the supplied key is invalid."""
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    if not x_api_key or not compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return True


class SlidingWindowRateLimiter:
    """Per-process, per-client sliding-window limiter for a single API instance."""

    def __init__(self, limit=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS):
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate-limit settings must be positive.")
        self.limit = limit
        self.window_seconds = window_seconds
        self._events = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_id, now=None):
        current = monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            # Bound memory if a large number of distinct client IPs appears.
            if len(self._events) >= _MAX_TRACKED_CLIENTS and client_id not in self._events:
                oldest_key = min(
                    self._events,
                    key=lambda key: self._events[key][0] if self._events[key] else current,
                )
                self._events.pop(oldest_key, None)
            events = self._events[client_id]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(events[0] + self.window_seconds - current))
                return False, retry_after
            events.append(current)
            return True, 0

    def reset(self):
        """Clear counters; intended for tests and controlled local maintenance."""
        with self._lock:
            self._events.clear()


rate_limiter = SlidingWindowRateLimiter()


def enforce_rate_limit(request: Request):
    """Limit authenticated prediction requests by the immediate peer address."""
    client_id = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.allow(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
    return True
