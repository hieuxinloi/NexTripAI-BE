from __future__ import annotations

import pytest
from anyio import fail_after, sleep

from src.apis.domains.chat.schemas import ChatResponse
from src.apis.domains.evaluations.manager import EvaluationManager
from src.apis.domains.evaluations.schemas import EvaluationJudgment
from src.apis.domains.evaluations.workbook import (
    EvaluationCaseInput,
    ParsedEvaluationWorkbook,
)
from src.infra.evaluation_store import InMemoryEvaluationStore


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeWorkerPool:
    async def submit(self, request, _kb_client, _answer_generator):
        return ChatResponse(
            session_id=request.session_id,
            message_id=f"answer-{request.session_id}",
            answer=f"Câu trả lời cho {request.message}",
            intent="test",
            kb_version=request.kb_version,
        )


class FakeJudge:
    model_name = "test-judge"

    def __init__(self) -> None:
        self.closed = False

    def evaluate(self, **_kwargs):
        return EvaluationJudgment(
            score=0.9,
            reason="Đáp ứng tiêu chí.",
        )

    def close(self) -> None:
        self.closed = True


class FakeChatStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_session(self, session_id: str, *, user_id: str) -> bool:
        self.deleted.append(f"{user_id}:{session_id}")
        return True


@pytest.mark.anyio
async def test_manager_runs_chat_and_judge_for_each_case() -> None:
    judge = FakeJudge()
    store = FakeChatStore()
    manager = EvaluationManager(
        worker_pool=FakeWorkerPool(),
        kb_client=object(),
        answer_generator=object(),
        chat_store=store,
        evaluation_store=InMemoryEvaluationStore(),
        judge=judge,
        chat_timeout_seconds=5,
    )
    workbook = ParsedEvaluationWorkbook(
        sheet_name="Test cases",
        cases=(
            EvaluationCaseInput(2, "Câu hỏi 1", "Kết quả 1"),
            EvaluationCaseInput(3, "Câu hỏi 2", "Kết quả 2"),
        ),
    )

    started = manager.start(
        workbook=workbook,
        filename="cases.xlsx",
        kb_version="v5",
        owner_id="user-a",
    )
    with fail_after(2):
        while (await manager.get(started.job_id, owner_id="user-a")).status != "completed":
            await sleep(0.01)
    result = await manager.get(started.job_id, owner_id="user-a")

    assert result.summary.total == 2
    assert result.summary.passed == 2
    assert result.summary.pass_rate == 1
    assert all(item.status == "passed" for item in result.cases)
    assert len(store.deleted) == 2
    history = await manager.list_history(owner_id="user-a", limit=10)
    assert history[0].job_id == result.job_id
    assert history[0].summary.passed == 2
    assert await manager.delete_history(result.job_id, owner_id="user-a") is True
    assert await manager.list_history(owner_id="user-a", limit=10) == []
    with pytest.raises(KeyError):
        await manager.get(result.job_id, owner_id="user-a")
    await manager.close()
    assert judge.closed is True
