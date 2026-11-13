"""API-key authentication and bounded per-process sliding-window rate limiting."""
from collections import defaultdict, deque
from hmac import compare_digest
import ipaddress
import os
from threading import Lock
from time import monotonic

from fastapi import Header, HTTPException, Request

API_KEY = os.getenv("DETECTOR_API_KEY", "")
RATE_LIMIT_REQUESTS = int(os.getenv("DETECTOR_RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("DETECTOR_RATE_LIMIT_WINDOW_SECONDS", "60"))
TRUSTED_PROXY_CIDRS = tuple(
    value.strip()
    for value in os.getenv("DETECTOR_TRUSTED_PROXY_CIDRS", "").split(",")
    if value.strip()
)
_MAX_TRACKED_CLIENTS = 10000


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Fail closed if no server key is configured or the supplied key is invalid."""
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    if not x_api_key or not compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return True


def resolve_client_identifier(peer_host, forwarded_for=None, trusted_proxy_cidrs=()):
    """Resolve a client IP only when the immediate peer is explicitly trusted.

    For a trusted reverse proxy that appends the connecting client to
    X-Forwarded-For, the right-most address is used. Untrusted peers cannot
    influence the identifier by supplying a forwarded header.
    """
    if not peer_host:
        return "unknown"

    try:
        peer_ip = ipaddress.ip_address(peer_host)
    except ValueError:
        return str(peer_host)

    trusted = False
    for value in trusted_proxy_cidrs:
        try:
            if peer_ip in ipaddress.ip_network(value, strict=False):
                trusted = True
                break
        except ValueError:
            # Invalid proxy configuration never grants trust.
            continue

    if trusted and forwarded_for:
        candidate = forwarded_for.split(",")[-1].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            # Malformed forwarded data falls back to the verified peer.
            pass

    return str(peer_ip)


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
    """Rate-limit by peer IP, or by forwarded client IP from a trusted proxy."""
    peer_host = request.client.host if request.client else None
    client_id = resolve_client_identifier(
        peer_host,
        request.headers.get("x-forwarded-for"),
        TRUSTED_PROXY_CIDRS,
    )
    allowed, retry_after = rate_limiter.allow(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
    return True
