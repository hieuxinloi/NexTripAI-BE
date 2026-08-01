from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.apis.domains.admin.router import _merge_firebase_users
from src.app import create_app
from src.core_ai.personalization.models import (
    PersonalizationProfile,
    PersonalizationUpdate,
    PreferenceEvent,
    SavedPlacesResponse,
)
from src.core_ai.personalization.recommendations import (
    RecommendationService,
    UnsupportedRecommendationVersionError,
    _current_taste,
)
from src.core_ai.personalization.service import compile_personalization_context
from src.infra.user_profile_store import (
    FirestoreUserProfileStore,
    InMemoryUserProfileStore,
    _firestore_document_id,
)
from src.infra.resilience import CircuitOpenError
from src.security.auth import Principal, current_principal


class FakeRecommendationKbClient:
    def __init__(self) -> None:
        self.request = None
        self.batch_requests: list[list[str]] = []

    def personalized_recommendations(self, **kwargs):
        self.request = kwargs
        return SavedPlacesResponse.model_validate({
            "items": [
                {
                    "place_id": "v8:beach-2",
                    "name": "Bãi biển 2",
                    "city": "Quy Nhơn",
                    "entity_type": "attraction",
                    "score": 0.8,
                    "reason_code": "similar_to_recent_place",
                    "reason": "Vì bạn từng quan tâm một bãi biển",
                }
            ]
        })

    def places_by_ids(self, place_ids, *, kb_version):
        self.batch_requests.append(place_ids)
        return SavedPlacesResponse.model_validate({
            "items": [
                {
                    "place_id": place_id,
                    "name": "Địa điểm đã lưu",
                    "city": "Quy Nhơn",
                    "entity_type": "attraction",
                    "reason": "Địa điểm bạn đã lưu",
                }
                for place_id in place_ids
            ]
        })


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
                "preferred_concepts": ["beach", "beach", "vietnamese"],
                "onboarding_completed": True,
                "onboarding_version": 1,
            },
        )
        assert response.status_code == 200
        assert response.json()["preferred_concepts"] == ["beach", "vietnamese"]

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
        app.state.user_profile_store = InMemoryUserProfileStore()
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


def test_disabled_personalization_keeps_explicit_recommendation_context() -> None:
    store = InMemoryUserProfileStore()
    store.update_profile(
        "alice",
        PersonalizationUpdate(
            personalization_enabled=False,
            preferred_cities=["Đà Nẵng"],
            preferred_concepts=["beach"],
        ),
    )
    kb_client = FakeRecommendationKbClient()

    feed = RecommendationService(store, kb_client, kb_version="v8").feed(
        "alice",
        city="Quy Nhơn",
        seed_place_ids=["v8:current-place"],
        limit=6,
    )

    assert feed.personalized is False
    assert feed.source == "disabled"
    assert kb_client.request["seed_place_ids"] == ["v8:current-place"]
    assert kb_client.request["preferred_cities"] == ["Quy Nhơn"]
    assert kb_client.request["preferred_concepts"] == []


def test_admin_requires_role_and_can_list_users(monkeypatch) -> None:
    from firebase_admin import auth

    empty_page = SimpleNamespace(iterate_all=lambda: iter(()))
    monkeypatch.setattr(auth, "list_users", lambda **_: empty_page)
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

        app.dependency_overrides[current_principal] = lambda: Principal(
            uid="firebase-admin-1",
            claims={"role": "admin"},
            auth_mode="firebase",
        )
        allowed = client.get("/api/admin/users")
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


def test_firestore_document_keys_do_not_embed_external_path_segments() -> None:
    key = _firestore_document_id("v8:source/place#1")

    assert len(key) == 64
    assert "/" not in key


def test_recommendation_feed_uses_recent_grounded_events_and_preferences() -> None:
    store = InMemoryUserProfileStore()
    store.update_profile(
        "alice",
        PersonalizationUpdate(
            preferred_concepts=["beach"],
            preferred_cities=["Quy Nhơn"],
        ),
    )
    store.save_event(
        "alice",
        PreferenceEvent(
            event_id="event-0001",
            event_type="ask_place",
            place_id="v8:beach-1",
        ),
    )
    kb_client = FakeRecommendationKbClient()

    feed = RecommendationService(store, kb_client, kb_version="v8").feed(
        "alice", city=None, seed_place_ids=[], limit=6
    )

    assert feed.personalized is True
    assert feed.items[0].place_id == "v8:beach-2"
    assert kb_client.request["seed_place_ids"] == ["v8:beach-1"]
    assert kb_client.request["preferred_concepts"] == ["beach"]
    assert "v8:beach-1" in kb_client.request["excluded_place_ids"]


def test_saved_places_follow_latest_save_state() -> None:
    store = InMemoryUserProfileStore()
    for event_id, event_type in [
        ("event-0001", "save"),
        ("event-0002", "unsave"),
        ("event-0003", "save"),
    ]:
        store.save_event(
            "alice",
            PreferenceEvent(
                event_id=event_id,
                event_type=event_type,
                place_id="v8:place-1",
            ),
        )

    result = RecommendationService(
        store, FakeRecommendationKbClient(), kb_version="v8"
    ).saved_places("alice")

    assert [item.place_id for item in result.items] == ["v8:place-1"]
    assert result.items[0].saved is True


def test_latest_preference_transition_controls_recommendation_state() -> None:
    store = InMemoryUserProfileStore()
    events = [
        ("event-0001", "like", "v8:place-1"),
        ("event-0002", "dislike", "v8:place-1"),
        ("event-0003", "like", "v8:place-1"),
        ("event-0004", "save", "v8:place-2"),
        ("event-0005", "unsave", "v8:place-2"),
        ("event-0006", "add_to_itinerary", "v8:place-3"),
        ("event-0007", "remove_from_itinerary", "v8:place-3"),
    ]
    for event_id, event_type, place_id in events:
        store.save_event(
            "alice",
            PreferenceEvent(
                event_id=event_id,
                event_type=event_type,
                place_id=place_id,
            ),
        )
    kb_client = FakeRecommendationKbClient()

    RecommendationService(store, kb_client, kb_version="v8").feed(
        "alice", city=None, seed_place_ids=[], limit=6
    )

    assert "v8:place-1" in kb_client.request["seed_place_ids"]
    assert "v8:place-2" not in kb_client.request["seed_place_ids"]
    assert "v8:place-3" not in kb_client.request["seed_place_ids"]


def test_duplicate_event_id_does_not_reapply_saved_state() -> None:
    store = InMemoryUserProfileStore()
    save = PreferenceEvent(
        event_id="event-0001",
        event_type="save",
        place_id="v8:place-1",
    )
    store.save_event("alice", save)
    store.save_event("alice", save.model_copy(update={"event_type": "unsave"}))

    assert store.saved_place_ids("alice") == ["v8:place-1"]


def test_recent_transient_events_are_prioritized_as_recommendation_seeds() -> None:
    store = InMemoryUserProfileStore()
    for index in range(25):
        store.save_event(
            "alice",
            PreferenceEvent(
                event_id=f"event-{index:04d}",
                event_type="ask_place",
                place_id=f"v8:place-{index}",
            ),
        )

    positive, _ = _current_taste(store.recent_events("alice"))

    assert positive[:3] == ["v8:place-24", "v8:place-23", "v8:place-22"]


def test_explicit_negative_feedback_overrides_passive_interest() -> None:
    store = InMemoryUserProfileStore()
    store.save_event(
        "alice",
        PreferenceEvent(
            event_id="event-0001",
            event_type="view_detail",
            place_id="v8:place-1",
        ),
    )
    store.save_event(
        "alice",
        PreferenceEvent(
            event_id="event-0002",
            event_type="dismiss_recommendation",
            place_id="v8:place-1",
        ),
    )

    positive, excluded = _current_taste(store.recent_events("alice"))

    assert "v8:place-1" not in positive
    assert excluded == ["v8:place-1"]


def test_recommendations_reject_an_unsupported_active_kb_version() -> None:
    with pytest.raises(UnsupportedRecommendationVersionError):
        RecommendationService(
            InMemoryUserProfileStore(),
            FakeRecommendationKbClient(),
            kb_version="v5",
        )


def test_recommendations_map_an_open_kb_circuit_to_retryable_service_error(
    monkeypatch,
) -> None:
    def fail_feed(*args, **kwargs):
        raise CircuitOpenError("Dependency circuit is open")

    monkeypatch.setattr(RecommendationService, "feed", fail_feed)
    app = create_app()

    with TestClient(app) as client:
        app.state.settings.active_kb_version = "v8"
        response = client.get(
            "/api/me/recommendations",
            headers={"X-Dev-User-ID": "alice"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Personalized recommendations are temporarily unavailable."
    )


def test_saved_places_are_split_into_kb_batches() -> None:
    store = InMemoryUserProfileStore()
    for index in range(101):
        store.save_event(
            "alice",
            PreferenceEvent(
                event_id=f"event-{index:04d}",
                event_type="save",
                place_id=f"v8:place-{index}",
            ),
        )
    kb_client = FakeRecommendationKbClient()

    result = RecommendationService(store, kb_client, kb_version="v8").saved_places(
        "alice"
    )

    assert len(result.items) == 101
    assert [len(batch) for batch in kb_client.batch_requests] == [100, 1]
