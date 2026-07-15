from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


class IdempotencyCoordinator:
    """Coalesces duplicate requests handled by the same application instance."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._inflight: dict[tuple[str, str], asyncio.Future[object]] = {}

    async def run(
        self,
        session_id: str,
        key: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        cache_key = (session_id, key)
        async with self._lock:
            existing = self._inflight.get(cache_key)
            if existing is None:
                future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
                self._inflight[cache_key] = future
                owner = True
            else:
                future = existing
                owner = False
        if not owner:
            return await asyncio.shield(future)  # type: ignore[return-value]
        try:
            result = await operation()
            future.set_result(result)
            return result
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                future.exception()
            raise
        finally:
            async with self._lock:
                self._inflight.pop(cache_key, None)
