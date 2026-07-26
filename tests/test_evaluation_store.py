from __future__ import annotations

from datetime import datetime, timezone

from src.apis.domains.evaluations.schemas import (
    EvaluationCaseResult,
    EvaluationJobResponse,
    EvaluationSummary,
)
from src.infra.evaluation_store import InMemoryEvaluationStore


def _job() -> EvaluationJobResponse:
    return EvaluationJobResponse(
        job_id="job-1",
        filename="cases.xlsx",
        sheet_name="Test cases",
        kb_version="v5",
        judge_model="test-judge",
        pass_threshold=0.8,
        status="running",
        summary=EvaluationSummary(
            total=1,
            completed=0,
            passed=0,
            failed=0,
            errors=0,
            pass_rate=0,
        ),
        cases=[
            EvaluationCaseResult(
                row_number=2,
                question="Câu hỏi",
                expected="Kết quả",
            )
        ],
        created_at=datetime.now(timezone.utc),
    )


def test_memory_evaluation_store_round_trips_job_and_case() -> None:
    store = InMemoryEvaluationStore()
    job = _job()
    store.save_job(job, owner_id="user-a")
    completed_case = job.cases[0].model_copy(
        update={
            "status": "passed",
            "passed": True,
            "score": 0.95,
            "actual_answer": "Câu trả lời",
        }
    )
    store.save_case("job-1", completed_case, owner_id="user-a")

    restored = store.get_job("job-1", owner_id="user-a")
    history = store.list_jobs(owner_id="user-a", limit=10)

    assert restored is not None
    assert restored.cases[0].status == "passed"
    assert history[0].job_id == "job-1"
    assert store.get_job("job-1", owner_id="user-b") is None
