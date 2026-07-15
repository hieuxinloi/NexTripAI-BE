from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException


class InMemoryRateLimiter:
    """Per-instance sliding-window limiter for app-level traffic shaping."""

    def __init__(self, requests: int, window_seconds: int) -> None:
        self._limit = requests
        self._window = float(window_seconds)
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._checks = 0

    def check(self, key: str) -> None:
        now = monotonic()
        cutoff = now - self._window
        with self._lock:
            self._checks += 1
            if self._checks % 100 == 0:
                stale = [
                    item_key
                    for item_key, values in self._requests.items()
                    if not values or values[-1] <= cutoff
                ]
                for item_key in stale:
                    self._requests.pop(item_key, None)
            timestamps = self._requests[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self._limit:
                retry_after = max(1, int(timestamps[0] + self._window - now) + 1)
                raise HTTPException(
                    status_code=429,
                    detail="Too many chat requests. Please retry shortly.",
                    headers={"Retry-After": str(retry_after)},
                )
            timestamps.append(now)
