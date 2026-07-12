from __future__ import annotations

from threading import Event as ThreadEvent
from threading import Lock

import pytest
from anyio import create_task_group, fail_after, sleep

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse
from src.apis.domains.chat.worker_pool import ChatWorkerPool


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_sixth_chat_waits_until_one_of_five_workers_finishes() -> None:
    release_workers = ThreadEvent()
    state_lock = Lock()
    active_jobs = 0
    maximum_active_jobs = 0

    def blocking_handler(request, _kb_client, _answer_generator):
        nonlocal active_jobs, maximum_active_jobs
        with state_lock:
            active_jobs += 1
            maximum_active_jobs = max(maximum_active_jobs, active_jobs)
        release_workers.wait(timeout=3)
        with state_lock:
            active_jobs -= 1
        return ChatResponse(
            answer=request.message,
            intent="test",
            kb_version=request.kb_version,
        )

    pool = ChatWorkerPool(
        worker_count=5,
        queue_capacity=10,
        handler=blocking_handler,
    )
    results: list[ChatResponse] = []

    async def submit(index: int) -> None:
        response = await pool.submit(
            ChatRequest(message=f"message-{index}", session_id=f"session-{index}"),
            None,
            None,
        )
        results.append(response)

    async with pool.run():
        async with create_task_group() as task_group:
            for index in range(6):
                task_group.start_soon(submit, index)

            with fail_after(2):
                while True:
                    statistics = pool.statistics()
                    if statistics["active_jobs"] == 5 and statistics["queued_jobs"] == 1:
                        break
                    await sleep(0.01)

            assert maximum_active_jobs == 5
            release_workers.set()

    assert len(results) == 6
    assert maximum_active_jobs == 5


@pytest.mark.anyio
async def test_failed_chat_does_not_stop_worker_pool() -> None:
    def handler(request, _kb_client, _answer_generator):
        if request.message == "fail":
            raise ValueError("failed job")
        return ChatResponse(
            answer=request.message,
            intent="test",
            kb_version=request.kb_version,
        )

    pool = ChatWorkerPool(worker_count=1, queue_capacity=2, handler=handler)

    async with pool.run():
        with pytest.raises(ValueError, match="failed job"):
            await pool.submit(
                ChatRequest(message="fail", session_id="failed-session"),
                None,
                None,
            )

        response = await pool.submit(
            ChatRequest(message="next", session_id="next-session"),
            None,
            None,
        )

    assert response.answer == "next"
