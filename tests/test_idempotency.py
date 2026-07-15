from __future__ import annotations

import asyncio

from src.apis.domains.chat.idempotency import IdempotencyCoordinator


def test_idempotency_coordinator_coalesces_inflight_requests() -> None:
    async def scenario() -> tuple[list[str], int]:
        coordinator = IdempotencyCoordinator()
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return "result"

        results = await asyncio.gather(
            coordinator.run("session", "request-key", operation),
            coordinator.run("session", "request-key", operation),
        )
        return results, calls

    results, calls = asyncio.run(scenario())

    assert results == ["result", "result"]
    assert calls == 1
