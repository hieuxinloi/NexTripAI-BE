from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import random
from threading import Lock
from time import monotonic, sleep
from typing import Generic, TypeVar


T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


@dataclass(slots=True)
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TtlCache(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, _CacheEntry[T]] = {}
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        now = monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item.expires_at <= now:
                self._items.pop(key, None)
                return None
            return item.value

    def set(self, key: str, value: T, ttl_seconds: float) -> None:
        with self._lock:
            self._items[key] = _CacheEntry(value, monotonic() + ttl_seconds)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_seconds: float) -> None:
        self._threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    def call(self, operation: Callable[[], T]) -> T:
        with self._lock:
            if self._opened_at is not None:
                elapsed = monotonic() - self._opened_at
                if elapsed < self._recovery_seconds:
                    raise CircuitOpenError("Dependency circuit is open")
                self._opened_at = None
                self._failures = 0
        try:
            result = operation()
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self._threshold:
                    self._opened_at = monotonic()
            raise
        with self._lock:
            self._failures = 0
        return result


def retry(
    operation: Callable[[], T],
    *,
    attempts: int,
    retryable: Callable[[Exception], bool],
    base_delay_seconds: float = 0.15,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not retryable(exc):
                raise
            sleep(base_delay_seconds * (2 ** (attempt - 1)) + random() * 0.05)
    raise AssertionError("unreachable")
