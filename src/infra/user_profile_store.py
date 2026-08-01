from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from threading import Lock
from typing import Any, Protocol

from src.config import Settings
from src.core_ai.personalization.models import (
    PersonalizationProfile,
    PersonalizationUpdate,
    PreferenceEvent,
    PreferenceEventRecord,
)


class ProfileRevisionConflictError(RuntimeError):
    pass


def _firestore_document_id(value: str) -> str:
    """Return a stable Firestore-safe key while keeping the original ID in data."""
    return sha256(value.encode("utf-8")).hexdigest()


class UserProfileStore(Protocol):
    backend_name: str

    def get_profile(self, user_id: str) -> PersonalizationProfile: ...
    def update_profile(
        self, user_id: str, update: PersonalizationUpdate
    ) -> PersonalizationProfile: ...
    def reset_profile(self, user_id: str) -> PersonalizationProfile: ...
    def save_event(self, user_id: str, event: PreferenceEvent) -> None: ...
    def recent_events(
        self, user_id: str, limit: int = 100
    ) -> list[PreferenceEventRecord]: ...
    def saved_place_ids(
        self, user_id: str, limit: int | None = None
    ) -> list[str]: ...
    def list_users(self, limit: int = 100) -> list[dict[str, Any]]: ...
    def update_admin_metadata(
        self,
        user_id: str,
        *,
        status: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]: ...
    def record_audit(
        self,
        *,
        actor_uid: str,
        action: str,
        target_uid: str,
        details: dict[str, Any] | None = None,
    ) -> None: ...
    def list_audit(self, limit: int = 100) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


class InMemoryUserProfileStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self._profiles: dict[str, PersonalizationProfile] = {}
        self._roles: dict[str, str] = {}
        self._events: dict[tuple[str, str], PreferenceEventRecord] = {}
        self._saved_places: dict[str, dict[str, datetime]] = {}
        self._audit: list[dict[str, Any]] = []
        self._lock = Lock()

    def get_profile(self, user_id: str) -> PersonalizationProfile:
        with self._lock:
            return self._profiles.get(
                user_id,
                PersonalizationProfile(user_id=user_id),
            ).model_copy(deep=True)

    def update_profile(
        self,
        user_id: str,
        update: PersonalizationUpdate,
    ) -> PersonalizationProfile:
        with self._lock:
            current = self._profiles.get(
                user_id,
                PersonalizationProfile(user_id=user_id),
            )
            if (
                update.expected_revision is not None
                and update.expected_revision != current.profile_revision
            ):
                raise ProfileRevisionConflictError("Profile revision is stale.")
            changes = update.model_dump(
                exclude_none=True,
                exclude={"expected_revision"},
            )
            now = datetime.now(timezone.utc)
            profile = current.model_copy(
                update={
                    **changes,
                    "profile_revision": current.profile_revision + 1,
                    "created_at": current.created_at or now,
                    "updated_at": now,
                }
            )
            self._profiles[user_id] = profile
            return profile.model_copy(deep=True)

    def reset_profile(self, user_id: str) -> PersonalizationProfile:
        with self._lock:
            current = self._profiles.get(
                user_id,
                PersonalizationProfile(user_id=user_id),
            )
            profile = PersonalizationProfile(
                user_id=user_id,
                personalization_enabled=current.personalization_enabled,
                profile_revision=current.profile_revision + 1,
                created_at=current.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            self._profiles[user_id] = profile
            return profile.model_copy(deep=True)

    def save_event(self, user_id: str, event: PreferenceEvent) -> None:
        with self._lock:
            key = (user_id, event.event_id)
            if key in self._events:
                return
            self._events[key] = PreferenceEventRecord(
                **event.model_dump(),
                created_at=datetime.now(timezone.utc),
            )
            if event.event_type == "save":
                self._saved_places.setdefault(user_id, {})[event.place_id] = (
                    datetime.now(timezone.utc)
                )
            elif event.event_type == "unsave":
                self._saved_places.setdefault(user_id, {}).pop(event.place_id, None)

    def recent_events(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[PreferenceEventRecord]:
        with self._lock:
            events = [
                event.model_copy(deep=True)
                for (owner_id, _), event in reversed(tuple(self._events.items()))
                if owner_id == user_id
            ]
        return sorted(
            events,
            key=lambda event: event.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )[:limit]

    def saved_place_ids(
        self,
        user_id: str,
        limit: int | None = None,
    ) -> list[str]:
        with self._lock:
            saved = self._saved_places.get(user_id, {})
            place_ids = [
                place_id
                for place_id, _ in sorted(
                    saved.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
        return place_ids[:limit] if limit is not None else place_ids

    def list_users(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    **profile.model_dump(mode="json"),
                    "role_display": self._roles.get(profile.user_id, "user"),
                }
                for profile in list(self._profiles.values())[:limit]
            ]

    def update_admin_metadata(
        self,
        user_id: str,
        *,
        status: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            current = self._profiles.get(
                user_id,
                PersonalizationProfile(user_id=user_id),
            )
            changes: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
            if status is not None:
                changes["status"] = status
            profile = current.model_copy(update=changes)
            self._profiles[user_id] = profile
            result = profile.model_dump(mode="json")
            if role is not None:
                self._roles[user_id] = role
                result["role_display"] = role
            return result

    def record_audit(
        self,
        *,
        actor_uid: str,
        action: str,
        target_uid: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._audit.append(
                {
                    "actor_uid": actor_uid,
                    "action": action,
                    "target_uid": target_uid,
                    "details": details or {},
                    "created_at": datetime.now(timezone.utc),
                }
            )

    def list_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in reversed(self._audit[-limit:])]

    def close(self) -> None:
        return None


class FirestoreUserProfileStore:
    backend_name = "firestore"

    def __init__(self, app_settings: Settings) -> None:
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise RuntimeError(
                "Firestore profile persistence requires google-cloud-firestore."
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
        self._audit = self._client.collection(app_settings.firestore_audit_collection)

    def get_profile(self, user_id: str) -> PersonalizationProfile:
        user_ref = self._users.document(user_id)
        preference_ref = user_ref.collection("preferences").document("main")
        snapshots = {
            snapshot.reference.path: snapshot
            for snapshot in self._client.get_all([user_ref, preference_ref])
        }
        user = snapshots[user_ref.path]
        preference = snapshots[preference_ref.path]
        data = user.to_dict() if user.exists else {}
        preference_data = preference.to_dict() if preference.exists else {}
        return PersonalizationProfile.model_validate(
            {"user_id": user_id, **(data or {}), **(preference_data or {})}
        )

    def update_profile(
        self,
        user_id: str,
        update: PersonalizationUpdate,
    ) -> PersonalizationProfile:
        changes = update.model_dump(exclude_none=True, exclude={"expected_revision"})
        metadata_keys = {
            "personalization_enabled",
            "onboarding_completed",
            "onboarding_version",
        }
        metadata = {
            key: value for key, value in changes.items() if key in metadata_keys
        }
        preferences = {
            key: value for key, value in changes.items() if key not in metadata_keys
        }
        user_ref = self._users.document(user_id)
        preference_ref = user_ref.collection("preferences").document("main")
        transaction = self._client.transaction()

        @self._firestore.transactional
        def commit_profile(transaction: Any) -> None:
            user_snapshot = user_ref.get(transaction=transaction)
            preference_snapshot = preference_ref.get(transaction=transaction)
            current = PersonalizationProfile.model_validate(
                {
                    "user_id": user_id,
                    **(user_snapshot.to_dict() if user_snapshot.exists else {}),
                    **(
                        preference_snapshot.to_dict()
                        if preference_snapshot.exists
                        else {}
                    ),
                }
            )
            if (
                update.expected_revision is not None
                and update.expected_revision != current.profile_revision
            ):
                raise ProfileRevisionConflictError("Profile revision is stale.")
            now = datetime.now(timezone.utc)
            transaction.set(
                user_ref,
                {
                    **metadata,
                    "profile_revision": current.profile_revision + 1,
                    "created_at": current.created_at or now,
                    "updated_at": now,
                },
                merge=True,
            )
            if preferences:
                transaction.set(
                    preference_ref,
                    {**preferences, "updated_at": now},
                    merge=True,
                )

        commit_profile(transaction)
        return self.get_profile(user_id)

    def reset_profile(self, user_id: str) -> PersonalizationProfile:
        current = self.get_profile(user_id)
        user_ref = self._users.document(user_id)
        preference_ref = user_ref.collection("preferences").document("main")
        batch = self._client.batch()
        batch.delete(preference_ref)
        batch.set(
            user_ref,
            {
                "onboarding_completed": False,
                "onboarding_version": 0,
                "profile_revision": current.profile_revision + 1,
                "updated_at": datetime.now(timezone.utc),
            },
            merge=True,
        )
        batch.commit()
        return self.get_profile(user_id)

    def save_event(self, user_id: str, event: PreferenceEvent) -> None:
        user_ref = self._users.document(user_id)
        event_ref = user_ref.collection("preference_events").document(
            _firestore_document_id(event.event_id)
        )
        saved_ref = user_ref.collection("saved_places").document(
            _firestore_document_id(event.place_id)
        )
        transaction = self._client.transaction()

        @self._firestore.transactional
        def commit_event(transaction: Any) -> None:
            if event_ref.get(transaction=transaction).exists:
                return
            transaction.create(
                event_ref,
                {
                    **event.model_dump(mode="json"),
                    "created_at": self._firestore.SERVER_TIMESTAMP,
                },
            )
            if event.event_type == "save":
                transaction.set(
                    saved_ref,
                    {
                        "place_id": event.place_id,
                        "saved_at": self._firestore.SERVER_TIMESTAMP,
                    },
                )
            elif event.event_type == "unsave":
                transaction.delete(saved_ref)

        commit_event(transaction)

    def recent_events(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[PreferenceEventRecord]:
        query = (
            self._users.document(user_id)
            .collection("preference_events")
            .order_by("created_at", direction=self._firestore.Query.DESCENDING)
            .limit(limit)
        )
        return [
            PreferenceEventRecord.model_validate(document.to_dict() or {})
            for document in query.stream()
        ]

    def saved_place_ids(
        self,
        user_id: str,
        limit: int | None = None,
    ) -> list[str]:
        query = (
            self._users.document(user_id)
            .collection("saved_places")
            .order_by("saved_at", direction=self._firestore.Query.DESCENDING)
        )
        if limit is not None:
            query = query.limit(limit)
        return [
            str((document.to_dict() or {}).get("place_id") or document.id)
            for document in query.stream()
        ]

    def list_users(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {"user_id": document.id, **(document.to_dict() or {})}
            for document in self._users.limit(limit).stream()
        ]

    def update_admin_metadata(
        self,
        user_id: str,
        *,
        status: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        changes: dict[str, Any] = {
            "updated_at": self._firestore.SERVER_TIMESTAMP,
        }
        if status is not None:
            changes["status"] = status
        if role is not None:
            changes["role_display"] = role
        reference = self._users.document(user_id)
        reference.set(changes, merge=True)
        result = self.get_profile(user_id).model_dump(mode="json")
        if role is not None:
            result["role_display"] = role
        return result

    def record_audit(
        self,
        *,
        actor_uid: str,
        action: str,
        target_uid: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._audit.document().set(
            {
                "actor_uid": actor_uid,
                "action": action,
                "target_uid": target_uid,
                "details": details or {},
                "created_at": self._firestore.SERVER_TIMESTAMP,
            }
        )

    def list_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        query = self._audit.order_by(
            "created_at",
            direction=self._firestore.Query.DESCENDING,
        ).limit(limit)
        return [
            {"event_id": document.id, **(document.to_dict() or {})}
            for document in query.stream()
        ]

    def close(self) -> None:
        self._client.close()


def create_user_profile_store(app_settings: Settings) -> UserProfileStore:
    if app_settings.chat_store_backend.strip().lower() == "firestore":
        return FirestoreUserProfileStore(app_settings)
    return InMemoryUserProfileStore()
