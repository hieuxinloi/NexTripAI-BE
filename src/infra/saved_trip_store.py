from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Protocol

from src.config import Settings


class SavedTripRevisionConflictError(RuntimeError):
    """Raised when a saved-trip update is based on a stale revision."""

    def __init__(
        self,
        *,
        trip_id: str,
        expected_revision: int,
        actual_revision: int,
    ) -> None:
        self.trip_id = trip_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            "Saved trip revision conflict: "
            f"expected {expected_revision}, current revision is {actual_revision}."
        )


class SavedTripStore(Protocol):
    backend_name: str

    def create(
        self,
        *,
        user_id: str,
        source_session_id: str,
        plan: dict[str, Any],
        title: str,
    ) -> dict[str, Any]: ...

    def list(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]: ...

    def get(self, trip_id: str, *, user_id: str) -> dict[str, Any] | None: ...

    def update(
        self,
        trip_id: str,
        *,
        user_id: str,
        expected_revision: int,
        title: str | None = None,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def delete(self, trip_id: str, *, user_id: str) -> bool: ...

    def close(self) -> None: ...


class InMemorySavedTripStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self._trips: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = Lock()

    def create(
        self,
        *,
        user_id: str,
        source_session_id: str,
        plan: dict[str, Any],
        title: str,
    ) -> dict[str, Any]:
        trip_id = _trip_id(plan)
        now = datetime.now(timezone.utc)
        with self._lock:
            user_trips = self._trips.setdefault(user_id, {})
            existing = user_trips.get(trip_id)
            if existing is not None:
                return deepcopy(existing)
            saved = {
                "trip_id": trip_id,
                "title": title,
                "source_session_id": source_session_id,
                "plan": deepcopy(plan),
                "revision": 1,
                "created_at": now,
                "updated_at": now,
            }
            user_trips[trip_id] = saved
            return deepcopy(saved)

    def list(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._trips.get(user_id, {}).values(),
                key=lambda item: item["updated_at"],
                reverse=True,
            )
            return deepcopy(items[:limit])

    def get(self, trip_id: str, *, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._trips.get(user_id, {}).get(trip_id)
            return deepcopy(item) if item is not None else None

    def update(
        self,
        trip_id: str,
        *,
        user_id: str,
        expected_revision: int,
        title: str | None = None,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            item = self._trips.get(user_id, {}).get(trip_id)
            if item is None:
                return None
            actual_revision = int(item["revision"])
            if actual_revision != expected_revision:
                raise SavedTripRevisionConflictError(
                    trip_id=trip_id,
                    expected_revision=expected_revision,
                    actual_revision=actual_revision,
                )
            if title is not None:
                item["title"] = title
            if plan is not None:
                item["plan"] = deepcopy(plan)
            item["revision"] = actual_revision + 1
            item["updated_at"] = datetime.now(timezone.utc)
            return deepcopy(item)

    def delete(self, trip_id: str, *, user_id: str) -> bool:
        with self._lock:
            user_trips = self._trips.get(user_id)
            if not user_trips or trip_id not in user_trips:
                return False
            del user_trips[trip_id]
            return True

    def close(self) -> None:
        return None


class FirestoreSavedTripStore:
    backend_name = "firestore"

    def __init__(self, app_settings: Settings) -> None:
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise RuntimeError(
                "Firestore saved-trip persistence requires google-cloud-firestore."
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
        self._users = self._client.collection(app_settings.firestore_users_collection)

    def _collection(self, user_id: str) -> Any:
        return self._users.document(user_id).collection("saved_trips")

    def create(
        self,
        *,
        user_id: str,
        source_session_id: str,
        plan: dict[str, Any],
        title: str,
    ) -> dict[str, Any]:
        trip_id = _trip_id(plan)
        reference = self._collection(user_id).document(_document_id(trip_id))
        transaction = self._client.transaction()

        @self._firestore.transactional
        def commit(transaction: Any) -> dict[str, Any]:
            snapshot = reference.get(transaction=transaction)
            if snapshot.exists:
                return {"trip_id": trip_id, **(snapshot.to_dict() or {})}
            now = datetime.now(timezone.utc)
            payload = {
                "trip_id": trip_id,
                "title": title,
                "source_session_id": source_session_id,
                "plan": deepcopy(plan),
                "revision": 1,
                "created_at": now,
                "updated_at": now,
            }
            transaction.create(reference, payload)
            return payload

        return deepcopy(commit(transaction))

    def list(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        query = self._collection(user_id).order_by(
            "updated_at",
            direction=self._firestore.Query.DESCENDING,
        ).limit(limit)
        return [
            {"trip_id": (document.to_dict() or {}).get("trip_id") or document.id,
             **(document.to_dict() or {})}
            for document in query.stream()
        ]

    def get(self, trip_id: str, *, user_id: str) -> dict[str, Any] | None:
        snapshot = self._collection(user_id).document(_document_id(trip_id)).get()
        if not snapshot.exists:
            return None
        return {"trip_id": trip_id, **(snapshot.to_dict() or {})}

    def update(
        self,
        trip_id: str,
        *,
        user_id: str,
        expected_revision: int,
        title: str | None = None,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        reference = self._collection(user_id).document(_document_id(trip_id))
        transaction = self._client.transaction()

        @self._firestore.transactional
        def commit(transaction: Any) -> dict[str, Any] | None:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                return None
            current = snapshot.to_dict() or {}
            actual_revision = int(current.get("revision") or 1)
            if actual_revision != expected_revision:
                raise SavedTripRevisionConflictError(
                    trip_id=trip_id,
                    expected_revision=expected_revision,
                    actual_revision=actual_revision,
                )
            changes: dict[str, Any] = {
                "revision": actual_revision + 1,
                "updated_at": datetime.now(timezone.utc),
            }
            if title is not None:
                changes["title"] = title
            if plan is not None:
                changes["plan"] = deepcopy(plan)
            transaction.set(reference, changes, merge=True)
            return {"trip_id": trip_id, **current, **changes}

        result = commit(transaction)
        return deepcopy(result) if result is not None else None

    def delete(self, trip_id: str, *, user_id: str) -> bool:
        reference = self._collection(user_id).document(_document_id(trip_id))
        snapshot = reference.get()
        if not snapshot.exists:
            return False
        reference.delete()
        return True

    def close(self) -> None:
        self._client.close()


def create_saved_trip_store(app_settings: Settings) -> SavedTripStore:
    if app_settings.chat_store_backend.strip().lower() == "firestore":
        return FirestoreSavedTripStore(app_settings)
    return InMemorySavedTripStore()


def _trip_id(plan: dict[str, Any]) -> str:
    plan_id = str(plan.get("plan_id") or "").strip()
    if not plan_id:
        raise ValueError("A saved trip requires a plan_id.")
    return plan_id[:128]


def _document_id(value: str) -> str:
    return value.replace("/", "_")[:128]
