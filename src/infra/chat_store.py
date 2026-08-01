from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
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

    def recent_messages(
        self,
        session_id: str,
        limit: int,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def create_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]: ...

    def list_sessions(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def rename_session(
        self,
        session_id: str,
        title: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None: ...

    def delete_session(self, session_id: str, *, user_id: str) -> bool: ...

    def get_session_memory(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]: ...

    def save_session_memory(
        self,
        session_id: str,
        memory: dict[str, Any],
        *,
        user_id: str | None = None,
    ) -> None: ...

    def get_idempotent_response(
        self,
        session_id: str,
        key: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None: ...

    def save_idempotent_response(
        self,
        session_id: str,
        key: str,
        response: dict[str, Any],
        *,
        user_id: str,
        ttl_seconds: int,
    ) -> None: ...

    def close(self) -> None: ...


class InMemoryChatStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._session_users: dict[str, str] = {}
        self._session_memory: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[tuple[str, str], dict[str, Any]] = {}
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
            self._claim_or_check(session_id, user_id)
            now = datetime.now(timezone.utc)
            session = self._sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "title": "Cuộc trò chuyện mới",
                    "created_at": now,
                    "updated_at": now,
                    "message_count": 0,
                    "user_id": user_id,
                },
            )
            if role == "user" and session["message_count"] == 0:
                session["title"] = content.strip().replace("\n", " ")[:120] or session["title"]
            session["updated_at"] = now
            session["message_count"] += 1
            self._messages[session_id].append(item)

    def recent_messages(
        self,
        session_id: str,
        limit: int,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._check_owner(session_id, user_id)
            return [dict(item) for item in self._messages.get(session_id, [])[-limit:]]

    def delete_session(self, session_id: str, *, user_id: str) -> bool:
        with self._lock:
            self._check_owner(session_id, user_id)
            existed = session_id in self._messages or session_id in self._session_users
            self._messages.pop(session_id, None)
            self._session_users.pop(session_id, None)
            self._session_memory.pop(session_id, None)
            self._sessions.pop(session_id, None)
            for cache_key in [key for key in self._idempotency if key[0] == session_id]:
                self._idempotency.pop(cache_key, None)
            return existed

    def create_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._check_owner(session_id, user_id)
            now = datetime.now(timezone.utc)
            session = self._sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "title": title or "Cuộc trò chuyện mới",
                    "created_at": now,
                    "updated_at": now,
                    "message_count": len(self._messages.get(session_id, [])),
                    "user_id": user_id,
                },
            )
            return dict(session)

    def list_sessions(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                dict(item)
                for item in reversed(tuple(self._sessions.values()))
                if user_id is None or item.get("user_id") in {None, user_id}
            ]
        rows.sort(key=lambda item: item.get("updated_at") or datetime.min, reverse=True)
        return rows[:limit]

    def rename_session(
        self,
        session_id: str,
        title: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            self._check_owner(session_id, user_id)
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session["title"] = title
            session["updated_at"] = datetime.now(timezone.utc)
            return dict(session)

    def get_session_memory(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._check_owner(session_id, user_id)
            return dict(self._session_memory.get(session_id, {}))

    def save_session_memory(
        self,
        session_id: str,
        memory: dict[str, Any],
        *,
        user_id: str | None = None,
    ) -> None:
        with self._lock:
            self._claim_or_check(session_id, user_id)
            self._session_memory[session_id] = dict(memory)

    def get_idempotent_response(
        self,
        session_id: str,
        key: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            self._check_owner(session_id, user_id)
            item = self._idempotency.get((session_id, key))
            if item is None:
                return None
            if item["expires_at"] <= datetime.now(timezone.utc):
                self._idempotency.pop((session_id, key), None)
                return None
            return dict(item["response"])

    def save_idempotent_response(
        self,
        session_id: str,
        key: str,
        response: dict[str, Any],
        *,
        user_id: str,
        ttl_seconds: int,
    ) -> None:
        with self._lock:
            self._claim_or_check(session_id, user_id)
            self._idempotency[(session_id, key)] = {
                "response": dict(response),
                "expires_at": datetime.now(timezone.utc)
                + timedelta(seconds=ttl_seconds),
            }

    def _claim_or_check(self, session_id: str, user_id: str | None) -> None:
        if user_id is None:
            return
        owner = self._session_users.get(session_id)
        if owner is not None and owner != user_id:
            raise PermissionError("Session belongs to another user")
        self._session_users.setdefault(session_id, user_id)

    def _check_owner(self, session_id: str, user_id: str | None) -> None:
        owner = self._session_users.get(session_id)
        if user_id is not None and owner is not None and owner != user_id:
            raise PermissionError("Session belongs to another user")

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
        credentials = None
        project = app_settings.google_cloud_project
        if app_settings.firestore_credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                app_settings.firestore_credentials_path
            )
            project = project or credentials.project_id
        self._client = firestore.Client(
            project=project,
            database=app_settings.firestore_database,
            credentials=credentials,
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
        self._check_owner(session_id, user_id)
        session_snapshot = session.get()
        existing = session_snapshot.to_dict() if session_snapshot.exists else {}
        message_count = int(existing.get("message_count") or 0) + 1
        session.set(
            {
                "user_id": user_id,
                "city": city,
                "title": (
                    existing.get("title")
                    or (content.strip().replace("\n", " ")[:120] if role == "user" else "Cuộc trò chuyện mới")
                ),
                "updated_at": self._firestore.SERVER_TIMESTAMP,
                "message_count": message_count,
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

    def create_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        session = self._sessions.document(session_id)
        self._check_owner(session_id, user_id)
        snapshot = session.get()
        if not snapshot.exists:
            session.set(
                {
                    "user_id": user_id,
                    "title": title or "Cuộc trò chuyện mới",
                    "created_at": self._firestore.SERVER_TIMESTAMP,
                    "updated_at": self._firestore.SERVER_TIMESTAMP,
                    "message_count": 0,
                },
                merge=True,
            )
        return {"session_id": session_id, "title": title or "Cuộc trò chuyện mới", "message_count": 0}

    def list_sessions(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = self._sessions.order_by("updated_at", direction=self._firestore.Query.DESCENDING).limit(limit)
        rows: list[dict[str, Any]] = []
        for document in query.stream():
            item = document.to_dict() or {}
            if user_id is not None and item.get("user_id") not in {None, user_id}:
                continue
            message_count = int(item.get("message_count") or 0)
            rows.append({
                "session_id": document.id,
                "title": item.get("title") or "Cuộc trò chuyện mới",
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "message_count": message_count,
            })
        return rows

    def recent_messages(
        self,
        session_id: str,
        limit: int,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._check_owner(session_id, user_id)
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

    def rename_session(
        self,
        session_id: str,
        title: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        session = self._sessions.document(session_id)
        snapshot = session.get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        self._check_owner(session_id, user_id, data)
        session.set(
            {
                "title": title,
                "updated_at": self._firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return {
            "session_id": session_id,
            "title": title,
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": int(data.get("message_count") or 0),
        }

    def delete_session(self, session_id: str, *, user_id: str) -> bool:
        session = self._sessions.document(session_id)
        snapshot = session.get()
        if not snapshot.exists:
            return False
        self._check_owner(session_id, user_id, snapshot.to_dict())
        self._delete_collection(session.collection("messages"))
        self._delete_collection(session.collection("idempotency"))
        session.delete()
        return True

    def get_session_memory(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._sessions.document(session_id)
        snapshot = session.get()
        if not snapshot.exists:
            return {}
        data = snapshot.to_dict() or {}
        self._check_owner(session_id, user_id, data)
        memory = data.get("memory")
        return dict(memory) if isinstance(memory, dict) else {}

    def save_session_memory(
        self,
        session_id: str,
        memory: dict[str, Any],
        *,
        user_id: str | None = None,
    ) -> None:
        session = self._sessions.document(session_id)
        self._check_owner(session_id, user_id)
        session.set(
            {
                "user_id": user_id,
                "memory": memory,
                "updated_at": self._firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    def get_idempotent_response(
        self,
        session_id: str,
        key: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        session = self._sessions.document(session_id)
        self._check_owner(session_id, user_id)
        snapshot = session.collection("idempotency").document(key).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        expires_at = data.get("expires_at")
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            snapshot.reference.delete()
            return None
        response = data.get("response")
        return dict(response) if isinstance(response, dict) else None

    def save_idempotent_response(
        self,
        session_id: str,
        key: str,
        response: dict[str, Any],
        *,
        user_id: str,
        ttl_seconds: int,
    ) -> None:
        session = self._sessions.document(session_id)
        self._check_owner(session_id, user_id)
        session.collection("idempotency").document(key).set(
            {
                "response": response,
                "expires_at": datetime.now(timezone.utc)
                + timedelta(seconds=ttl_seconds),
                "created_at": self._firestore.SERVER_TIMESTAMP,
            }
        )

    def _check_owner(
        self,
        session_id: str,
        user_id: str | None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if user_id is None:
            return
        if data is None:
            snapshot = self._sessions.document(session_id).get()
            if not snapshot.exists:
                return
            data = snapshot.to_dict() or {}
        owner = data.get("user_id")
        if owner is not None and owner != user_id:
            raise PermissionError("Session belongs to another user")

    @staticmethod
    def _delete_collection(collection: Any, batch_size: int = 100) -> None:
        while True:
            documents = list(collection.limit(batch_size).stream())
            if not documents:
                return
            for document in documents:
                document.reference.delete()

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
