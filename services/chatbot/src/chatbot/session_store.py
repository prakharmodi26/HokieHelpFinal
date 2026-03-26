"""Thread-safe in-memory session store with sliding-window rate limiting."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque


class SessionStore:
    """Per-session sliding-window rate limiter.

    Each session gets a deque of timestamps for requests within the current window.
    Requests older than window_seconds are evicted on each call to is_allowed().
    Thread-safe via a single reentrant lock.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._sessions: dict[str, deque[float]] = {}
        self._lock = threading.RLock()

    def get_or_create_session(self, session_id: str | None) -> str:
        """Return existing session_id or mint a new UUID."""
        if session_id and session_id.strip():
            return session_id.strip()
        return str(uuid.uuid4())

    def is_allowed(self, session_id: str) -> bool:
        """Record a request attempt and return True if within the rate limit."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque()
            q = self._sessions[session_id]
            # Evict timestamps outside the window
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._max:
                return False
            q.append(now)
            return True

    def remaining(self, session_id: str) -> int:
        """Return how many requests remain in the current window."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            q = self._sessions.get(session_id, deque())
            active = sum(1 for t in q if t >= cutoff)
            return max(0, self._max - active)
