from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from time import perf_counter
from typing import Any
from uuid import uuid4

from anyio import fail_after, to_thread
from loguru import logger

from src.apis.domains.chat.schemas import (
    AuthenticatedChatRequest,
    ChatResponse,
)
from src.apis.domains.chat.worker_pool import ChatWorkerPool
from src.apis.domains.evaluations.schemas import (
    EvaluationCaseResult,
    EvaluationHistoryItem,
    EvaluationJobResponse,
    EvaluationSummary,
)
from src.apis.domains.evaluations.workbook import ParsedEvaluationWorkbook
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
from src.infra.chat_store import ChatStore
from src.infra.evaluation_judge import SupportsEvaluationJudging
from src.infra.evaluation_store import EvaluationStore
from src.infra.kb_client import KbClient


PASS_THRESHOLD = 0.8
MAX_RETAINED_JOBS = 20


class EvaluationAlreadyRunningError(RuntimeError):
    pass


@dataclass(slots=True)
class _EvaluationJob:
    job_id: str
    owner_id: str
    filename: str
    sheet_name: str
    kb_version: str
    judge_model: str
    status: str
    cases: list[EvaluationCaseResult]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class EvaluationManager:
    def __init__(
        self,
        *,
        worker_pool: ChatWorkerPool,
        kb_client: KbClient,
        answer_generator: SupportsAnswerGeneration | None,
        chat_store: ChatStore,
        evaluation_store: EvaluationStore,
        judge: SupportsEvaluationJudging | None,
        chat_timeout_seconds: float,
        concurrency: int = 2,
    ) -> None:
        self._worker_pool = worker_pool
        self._kb_client = kb_client
        self._answer_generator = answer_generator
        self._chat_store = chat_store
        self._evaluation_store = evaluation_store
        self._judge = judge
        self._chat_timeout_seconds = chat_timeout_seconds
        self._concurrency = max(1, concurrency)
        self._jobs: dict[str, _EvaluationJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def available(self) -> bool:
        return self._judge is not None and self._answer_generator is not None

    @property
    def persistence_backend(self) -> str:
        return self._evaluation_store.backend_name

    def start(
        self,
        *,
        workbook: ParsedEvaluationWorkbook,
        filename: str,
        kb_version: str,
        owner_id: str,
    ) -> EvaluationJobResponse:
        if not self.available or self._judge is None:
            raise RuntimeError(
                "Evaluation requires Gemini answer generation and GOOGLE_API_KEY."
            )
        if any(
            job.owner_id == owner_id and job.status in {"queued", "running"}
            for job in self._jobs.values()
        ):
            raise EvaluationAlreadyRunningError(
                "Bạn đang có một lượt đánh giá chưa hoàn tất."
            )
        self._evict_old_jobs()
        job_id = uuid4().hex
        job = _EvaluationJob(
            job_id=job_id,
            owner_id=owner_id,
            filename=filename,
            sheet_name=workbook.sheet_name,
            kb_version=kb_version,
            judge_model=self._judge.model_name,
            status="queued",
            cases=[
                EvaluationCaseResult(
                    row_number=item.row_number,
                    question=item.question,
                    expected=item.expected,
                )
                for item in workbook.cases
            ],
            created_at=_now(),
        )
        self._jobs[job_id] = job
        task = asyncio.create_task(
            self._run_job(job),
            name=f"evaluation-{job_id}",
        )
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return self._snapshot(job)

    async def get(self, job_id: str, *, owner_id: str) -> EvaluationJobResponse:
        job = self._jobs.get(job_id)
        if job is not None:
            if job.owner_id != owner_id:
                raise PermissionError(job_id)
            return self._snapshot(job)
        persisted = await to_thread.run_sync(
            partial(
                self._evaluation_store.get_job,
                job_id,
                owner_id=owner_id,
            ),
            abandon_on_cancel=True,
        )
        if persisted is None:
            raise KeyError(job_id)
        return persisted

    async def list_history(
        self,
        *,
        owner_id: str,
        limit: int,
    ) -> list[EvaluationHistoryItem]:
        persisted = await to_thread.run_sync(
            partial(
                self._evaluation_store.list_jobs,
                owner_id=owner_id,
                limit=limit,
            ),
            abandon_on_cancel=True,
        )
        combined = {item.job_id: item for item in persisted}
        for job in self._jobs.values():
            if job.owner_id == owner_id:
                snapshot = self._snapshot(job)
                combined[job.job_id] = EvaluationHistoryItem.model_validate(
                    snapshot.model_dump(mode="python", exclude={"cases"})
                )
        return sorted(
            combined.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )[:limit]

    async def cancel(
        self,
        job_id: str,
        *,
        owner_id: str,
    ) -> EvaluationJobResponse:
        job = self._jobs.get(job_id)
        if job is None:
            return await self.get(job_id, owner_id=owner_id)
        if job.owner_id != owner_id:
            raise PermissionError(job_id)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        if job.status in {"queued", "running"}:
            job.status = "cancelled"
            job.completed_at = _now()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        return self._snapshot(job)

    async def close(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._judge is not None:
            self._judge.close()
        await to_thread.run_sync(
            self._evaluation_store.close,
            abandon_on_cancel=False,
        )

    async def _run_job(self, job: _EvaluationJob) -> None:
        job.status = "running"
        job.started_at = _now()
        semaphore = asyncio.Semaphore(self._concurrency)
        try:
            await self._persist_job(job, include_cases=True)
            await asyncio.gather(
                *(
                    self._run_case(job, case, semaphore)
                    for case in job.cases
                )
            )
            if job.status != "cancelled":
                job.status = "completed"
                job.completed_at = _now()
                await self._persist_job(job)
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.completed_at = _now()
            for case in job.cases:
                if case.status in {"pending", "running"}:
                    case.status = "error"
                    case.error = "Đã hủy đánh giá."
            await self._persist_job(job)
            raise
        except Exception as exc:
            logger.exception(
                "Evaluation job failed job_id={} error_type={}",
                job.job_id,
                exc.__class__.__name__,
            )
            job.status = "failed"
            job.error = "Không thể hoàn tất lượt đánh giá."
            job.completed_at = _now()
            await self._persist_job(job)

    async def _run_case(
        self,
        job: _EvaluationJob,
        case: EvaluationCaseResult,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            if job.status == "cancelled":
                return
            started_at = perf_counter()
            case.status = "running"
            session_id = f"eval-{job.job_id[:20]}-{case.row_number}"
            request = AuthenticatedChatRequest(
                message=case.question,
                session_id=session_id,
                kb_version=job.kb_version,
                user_id=job.owner_id,
            )
            try:
                with fail_after(self._chat_timeout_seconds):
                    response = await self._worker_pool.submit(
                        request,
                        self._kb_client,
                        self._answer_generator,
                    )
                case.actual_answer = response.answer
                if self._judge is None:
                    raise RuntimeError("Evaluation judge is unavailable.")
                judgment = await to_thread.run_sync(
                    partial(
                        self._judge.evaluate,
                        question=case.question,
                        expected=case.expected,
                        actual_answer=response.answer,
                        grounded_context=_grounded_context(response),
                    ),
                    abandon_on_cancel=False,
                )
                case.score = judgment.score
                case.passed = judgment.score >= PASS_THRESHOLD
                case.status = "passed" if case.passed else "failed"
                case.reason = judgment.reason
            except TimeoutError:
                case.status = "error"
                case.error = "Chat xử lý quá thời gian cho phép."
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Evaluation case failed job_id={} row={} error_type={}",
                    job.job_id,
                    case.row_number,
                    exc.__class__.__name__,
                )
                case.status = "error"
                case.error = str(exc)[:500] or exc.__class__.__name__
            finally:
                case.elapsed_ms = int((perf_counter() - started_at) * 1000)
                await self._persist_case(job, case)
                await self._delete_session(session_id, job.owner_id)

    async def _delete_session(self, session_id: str, owner_id: str) -> None:
        try:
            await to_thread.run_sync(
                partial(
                    self._chat_store.delete_session,
                    session_id,
                    user_id=owner_id,
                ),
                abandon_on_cancel=False,
            )
        except Exception as exc:
            logger.warning(
                "Evaluation session cleanup failed session_id={} error_type={}",
                session_id,
                exc.__class__.__name__,
            )

    def _evict_old_jobs(self) -> None:
        if len(self._jobs) < MAX_RETAINED_JOBS:
            return
        terminal = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in {"completed", "cancelled", "failed"}
            ),
            key=lambda item: item.created_at,
        )
        while len(self._jobs) >= MAX_RETAINED_JOBS and terminal:
            removed = terminal.pop(0)
            self._jobs.pop(removed.job_id, None)

    def _snapshot(self, job: _EvaluationJob) -> EvaluationJobResponse:
        completed = [
            case
            for case in job.cases
            if case.status in {"passed", "failed", "error"}
        ]
        passed = sum(case.status == "passed" for case in completed)
        failed = sum(case.status == "failed" for case in completed)
        errors = sum(case.status == "error" for case in completed)
        judged = passed + failed
        summary = EvaluationSummary(
            total=len(job.cases),
            completed=len(completed),
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=(passed / judged) if judged else 0,
        )
        return EvaluationJobResponse(
            job_id=job.job_id,
            filename=job.filename,
            sheet_name=job.sheet_name,
            kb_version=job.kb_version,
            judge_model=job.judge_model,
            pass_threshold=PASS_THRESHOLD,
            status=job.status,
            summary=summary,
            cases=[case.model_copy(deep=True) for case in job.cases],
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
        )

    async def _persist_job(
        self,
        job: _EvaluationJob,
        *,
        include_cases: bool = False,
    ) -> None:
        snapshot = self._snapshot(job)
        if not include_cases:
            snapshot.cases = []
        try:
            await to_thread.run_sync(
                partial(
                    self._evaluation_store.save_job,
                    snapshot,
                    owner_id=job.owner_id,
                ),
                abandon_on_cancel=False,
            )
        except Exception as exc:
            logger.exception(
                "Evaluation job persistence failed job_id={} error_type={}",
                job.job_id,
                exc.__class__.__name__,
            )

    async def _persist_case(
        self,
        job: _EvaluationJob,
        case: EvaluationCaseResult,
    ) -> None:
        try:
            await to_thread.run_sync(
                partial(
                    self._evaluation_store.save_case,
                    job.job_id,
                    case.model_copy(deep=True),
                    owner_id=job.owner_id,
                ),
                abandon_on_cancel=False,
            )
        except Exception as exc:
            logger.exception(
                "Evaluation case persistence failed job_id={} row={} error_type={}",
                job.job_id,
                case.row_number,
                exc.__class__.__name__,
            )


def _grounded_context(response: ChatResponse) -> dict[str, Any]:
    return {
        "intent": response.intent,
        "facts": response.facts[:30],
        "evidence": [
            {
                "name": item.name,
                "city": item.city,
                "entity_type": item.entity_type,
                "category": item.category,
                "attributes": item.attributes,
            }
            for item in response.evidence[:20]
        ],
        "missing_fields": response.missing_fields,
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)
