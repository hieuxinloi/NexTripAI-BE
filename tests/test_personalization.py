from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.apis.domains.admin.router import _merge_firebase_users
from src.app import create_app
from src.core_ai.personalization.models import PersonalizationProfile
from src.core_ai.personalization.service import compile_personalization_context
from src.infra.user_profile_store import FirestoreUserProfileStore


def test_user_can_update_and_reset_own_preferences() -> None:
    app = create_app()
    headers = {"X-Dev-User-ID": "alice"}
    with TestClient(app) as client:
        response = client.patch(
            "/api/me/preferences",
            headers=headers,
            json={
                "party_type": "family",
                "travel_pace": "relaxed",
                "preferred_concepts": ["beach", "beach", "local_food"],
                "onboarding_completed": True,
                "onboarding_version": 1,
            },
        )
        assert response.status_code == 200
        assert response.json()["preferred_concepts"] == ["beach", "local_food"]

        profile = client.get("/api/me/preferences", headers=headers)
        assert profile.status_code == 200
        assert profile.json()["party_type"] == "family"

        reset = client.delete("/api/me/preferences", headers=headers)
        assert reset.status_code == 200
        assert reset.json()["preferred_concepts"] == []


def test_stale_profile_revision_is_reported_as_conflict() -> None:
    app = create_app()
    headers = {"X-Dev-User-ID": "revision-user"}
    with TestClient(app) as client:
        first = client.patch(
            "/api/me/preferences",
            headers=headers,
            json={"party_type": "family"},
        )
        stale = client.patch(
            "/api/me/preferences",
            headers=headers,
            json={"budget_level": "premium", "expected_revision": 0},
        )

    assert first.status_code == 200
    assert first.json()["profile_revision"] == 1
    assert stale.status_code == 409
    assert stale.json()["detail"] == "Profile revision is stale."


def test_disabled_personalization_compiles_to_an_empty_chat_contract() -> None:
    profile = PersonalizationProfile(
        user_id="quiet-user",
        personalization_enabled=False,
        preferred_concepts=["beach"],
        preferred_cities=["Quy Nhon"],
    )

    assert compile_personalization_context(profile) == {}


def test_admin_requires_role_and_can_list_users() -> None:
    app = create_app()
    with TestClient(app) as client:
        client.patch(
            "/api/me/preferences",
            headers={"X-Dev-User-ID": "alice"},
            json={"party_type": "couple"},
        )
        denied = client.get(
            "/api/admin/users",
            headers={"X-Dev-User-ID": "alice"},
        )
        assert denied.status_code == 403

        allowed = client.get(
            "/api/admin/users",
            headers={"X-Dev-User-ID": "operator", "X-Dev-Role": "admin"},
        )
        assert allowed.status_code == 200
        assert allowed.json()["users"][0]["user_id"] == "alice"


def test_admin_user_list_merges_firebase_accounts(monkeypatch) -> None:
    from firebase_admin import auth

    accounts = [
        SimpleNamespace(
            uid="alice",
            email="alice@example.com",
            display_name="Alice",
            disabled=False,
            custom_claims={"role": "support"},
        ),
        SimpleNamespace(
            uid="new-user",
            email="new@example.com",
            display_name=None,
            disabled=True,
            custom_claims=None,
        ),
    ]
    page = SimpleNamespace(iterate_all=lambda: iter(accounts))
    monkeypatch.setattr(auth, "list_users", lambda **_: page)

    users = _merge_firebase_users(
        [{"user_id": "alice", "onboarding_completed": True}],
        limit=10,
    )

    assert users[0]["role_display"] == "support"
    assert users[0]["onboarding_completed"] is True
    assert users[1]["status"] == "suspended"


def test_firestore_profile_uses_one_batch_read() -> None:
    store = FirestoreUserProfileStore.__new__(FirestoreUserProfileStore)
    user_ref = MagicMock(path="users/alice")
    preference_ref = MagicMock(path="users/alice/preferences/main")
    user_ref.collection.return_value.document.return_value = preference_ref
    store._users = MagicMock()
    store._users.document.return_value = user_ref
    user_snapshot = SimpleNamespace(
        reference=user_ref,
        exists=True,
        to_dict=lambda: {"profile_revision": 2},
    )
    preference_snapshot = SimpleNamespace(
        reference=preference_ref,
        exists=True,
        to_dict=lambda: {"party_type": "family"},
    )
    store._client = MagicMock()
    store._client.get_all.return_value = [preference_snapshot, user_snapshot]

    profile = store.get_profile("alice")

    assert profile.profile_revision == 2
    assert profile.party_type == "family"
    store._client.get_all.assert_called_once_with([user_ref, preference_ref])


def test_firestore_profile_reset_commits_one_write_batch() -> None:
    store = FirestoreUserProfileStore.__new__(FirestoreUserProfileStore)
    user_ref = MagicMock()
    preference_ref = MagicMock()
    user_ref.collection.return_value.document.return_value = preference_ref
    store._users = MagicMock()
    store._users.document.return_value = user_ref
    batch = MagicMock()
    store._client = MagicMock()
    store._client.batch.return_value = batch
    current = PersonalizationProfile(user_id="alice", profile_revision=2)
    reset = PersonalizationProfile(user_id="alice", profile_revision=3)
    store.get_profile = MagicMock(side_effect=[current, reset])

    result = store.reset_profile("alice")

    assert result.profile_revision == 3
    batch.delete.assert_called_once_with(preference_ref)
    batch.set.assert_called_once()
    batch.commit.assert_called_once_with()
