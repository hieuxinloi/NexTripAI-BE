from __future__ import annotations

from hashlib import sha256
from threading import Lock
from typing import Any, Protocol

from src.apis.domains.evaluations.schemas import (
    EvaluationCaseResult,
    EvaluationHistoryItem,
    EvaluationJobResponse,
)
from src.config import Settings


EVALUATION_USERS_COLLECTION = "evaluation_users"
EVALUATION_RUNS_COLLECTION = "runs"
EVALUATION_CASES_COLLECTION = "cases"


class EvaluationStore(Protocol):
    backend_name: str

    def save_job(self, job: EvaluationJobResponse, *, owner_id: str) -> None: ...

    def save_case(
        self,
        job_id: str,
        case: EvaluationCaseResult,
        *,
        owner_id: str,
    ) -> None: ...

    def get_job(
        self,
        job_id: str,
        *,
        owner_id: str,
    ) -> EvaluationJobResponse | None: ...

    def list_jobs(
        self,
        *,
        owner_id: str,
        limit: int,
    ) -> list[EvaluationHistoryItem]: ...

    def close(self) -> None: ...


class InMemoryEvaluationStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], EvaluationJobResponse] = {}
        self._lock = Lock()

    def save_job(self, job: EvaluationJobResponse, *, owner_id: str) -> None:
        with self._lock:
            key = (owner_id, job.job_id)
            existing = self._jobs.get(key)
            saved = job.model_copy(deep=True)
            if existing is not None and not saved.cases:
                saved.cases = [case.model_copy(deep=True) for case in existing.cases]
            self._jobs[key] = saved

    def save_case(
        self,
        job_id: str,
        case: EvaluationCaseResult,
        *,
        owner_id: str,
    ) -> None:
        with self._lock:
            job = self._jobs.get((owner_id, job_id))
            if job is None:
                return
            cases = [
                case.model_copy(deep=True) if item.row_number == case.row_number else item
                for item in job.cases
            ]
            job.cases = cases

    def get_job(
        self,
        job_id: str,
        *,
        owner_id: str,
    ) -> EvaluationJobResponse | None:
        with self._lock:
            job = self._jobs.get((owner_id, job_id))
            return job.model_copy(deep=True) if job is not None else None

    def list_jobs(
        self,
        *,
        owner_id: str,
        limit: int,
    ) -> list[EvaluationHistoryItem]:
        with self._lock:
            jobs = sorted(
                (
                    job
                    for (stored_owner, _), job in self._jobs.items()
                    if stored_owner == owner_id
                ),
                key=lambda item: item.created_at,
                reverse=True,
            )[:limit]
            return [_history_item(job) for job in jobs]

    def close(self) -> None:
        return None


class FirestoreEvaluationStore:
    backend_name = "firestore"

    def __init__(self, app_settings: Settings):
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise RuntimeError(
                "Firestore evaluation persistence requires google-cloud-firestore."
            ) from exc
        credentials = None
        project = app_settings.google_cloud_project
        if app_settings.firestore_credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                app_settings.firestore_credentials_path
            )
            project = project or credentials.project_id
        self._firestore = firestore
        self._client = firestore.Client(
            project=project,
            database=app_settings.firestore_database,
            credentials=credentials,
        )

    def save_job(self, job: EvaluationJobResponse, *, owner_id: str) -> None:
        run = self._run_document(owner_id, job.job_id)
        run.set(_job_document(job), merge=True)
        if not job.cases:
            return
        for start in range(0, len(job.cases), 400):
            batch = self._client.batch()
            for case in job.cases[start : start + 400]:
                document = run.collection(EVALUATION_CASES_COLLECTION).document(
                    str(case.row_number)
                )
                batch.set(document, case.model_dump(mode="json"))
            batch.commit()

    def save_case(
        self,
        job_id: str,
        case: EvaluationCaseResult,
        *,
        owner_id: str,
    ) -> None:
        self._run_document(owner_id, job_id).collection(
            EVALUATION_CASES_COLLECTION
        ).document(str(case.row_number)).set(case.model_dump(mode="json"))

    def get_job(
        self,
        job_id: str,
        *,
        owner_id: str,
    ) -> EvaluationJobResponse | None:
        run = self._run_document(owner_id, job_id)
        snapshot = run.get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        cases = [
            EvaluationCaseResult.model_validate(item.to_dict() or {})
            for item in run.collection(EVALUATION_CASES_COLLECTION)
            .order_by("row_number")
            .stream()
        ]
        return EvaluationJobResponse.model_validate(data | {"cases": cases})

    def list_jobs(
        self,
        *,
        owner_id: str,
        limit: int,
    ) -> list[EvaluationHistoryItem]:
        runs = (
            self._runs_collection(owner_id)
            .order_by(
                "created_at",
                direction=self._firestore.Query.DESCENDING,
            )
            .limit(limit)
            .stream()
        )
        return [
            EvaluationHistoryItem.model_validate(snapshot.to_dict() or {})
            for snapshot in runs
        ]

    def close(self) -> None:
        self._client.close()

    def _runs_collection(self, owner_id: str):
        return (
            self._client.collection(EVALUATION_USERS_COLLECTION)
            .document(_owner_document_id(owner_id))
            .collection(EVALUATION_RUNS_COLLECTION)
        )

    def _run_document(self, owner_id: str, job_id: str):
        return self._runs_collection(owner_id).document(job_id)


def create_evaluation_store(app_settings: Settings) -> EvaluationStore:
    if app_settings.chat_store_backend == "memory":
        return InMemoryEvaluationStore()
    if app_settings.chat_store_backend == "firestore":
        return FirestoreEvaluationStore(app_settings)
    raise ValueError("CHAT_STORE_BACKEND must be either 'memory' or 'firestore'")


def _job_document(job: EvaluationJobResponse) -> dict[str, Any]:
    return job.model_dump(mode="python", exclude={"cases"})


def _history_item(job: EvaluationJobResponse) -> EvaluationHistoryItem:
    return EvaluationHistoryItem.model_validate(
        job.model_dump(mode="python", exclude={"cases"})
    )


def _owner_document_id(owner_id: str) -> str:
    return sha256(owner_id.encode("utf-8")).hexdigest()
