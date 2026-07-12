from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Protocol

from src.config import Settings


class ChatStore(Protocol):
    backend_name: str

    def save_message(
        self,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        *,
        user_id: str | None = None,
        city: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


class InMemoryChatStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = Lock()

    def save_message(
        self,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        *,
        user_id: str | None = None,
        city: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        item = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "user_id": user_id,
            "city": city,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
        }
        with self._lock:
            self._messages[session_id].append(item)

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._messages.get(session_id, [])[-limit:]]

    def close(self) -> None:
        return None


class FirestoreChatStore:
    backend_name = "firestore"

    def __init__(self, app_settings: Settings) -> None:
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise RuntimeError(
                "CHAT_STORE_BACKEND=firestore requires google-cloud-firestore."
            ) from exc
        self._firestore = firestore
        self._client = firestore.Client(
            project=app_settings.google_cloud_project,
            database=app_settings.firestore_database,
        )
        self._sessions = self._client.collection(
            app_settings.firestore_sessions_collection
        )

    def save_message(
        self,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        *,
        user_id: str | None = None,
        city: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        session = self._sessions.document(session_id)
        session.set(
            {
                "user_id": user_id,
                "city": city,
                "updated_at": self._firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        session.collection("messages").document(message_id).set(
            {
                "role": role,
                "content": content,
                "city": city,
                "metadata": metadata or {},
                "created_at": self._firestore.SERVER_TIMESTAMP,
            }
        )

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        query = (
            self._sessions.document(session_id)
            .collection("messages")
            .order_by("created_at", direction=self._firestore.Query.DESCENDING)
            .limit(limit)
        )
        rows = [
            document.to_dict() | {"message_id": document.id}
            for document in query.stream()
        ]
        rows.reverse()
        return rows

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


def create_chat_store(app_settings: Settings) -> ChatStore:
    backend = app_settings.chat_store_backend.strip().lower()
    if backend == "memory":
        return InMemoryChatStore()
    if backend == "firestore":
        return FirestoreChatStore(app_settings)
    raise ValueError(
        "CHAT_STORE_BACKEND must be either 'memory' or 'firestore', "
        f"got {app_settings.chat_store_backend!r}."
    )
