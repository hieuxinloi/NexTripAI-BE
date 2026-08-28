from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from src.app import create_app
from src.infra.saved_trip_store import (
    InMemorySavedTripStore,
    SavedTripRevisionConflictError,
)


def _plan() -> dict:
    return {
        "plan_id": "plan-qn-1",
        "revision": 1,
        "status": "complete",
        "city": "Quy Nhơn",
        "start_date": "2026-08-30",
        "duration_days": 1,
        "constraints": {"party_size": 2},
        "itinerary": [
            {
                "day": 1,
                "slots": [
                    {
                        "slot_id": "slot-1",
                        "order": 1,
                        "start_time": "08:00",
                        "end_time": "10:00",
                        "place_id": "attr_qn_001",
                        "name": "Kỳ Co",
                        "city": "Quy Nhơn",
                        "entity_type": "attraction",
                    }
                ],
            }
        ],
        "selected_places": [],
        "budget_summary": None,
        "last_operation": "create",
        "created_at": "2026-08-28T12:00:00Z",
        "updated_at": "2026-08-28T12:00:00Z",
    }


def test_memory_saved_trip_store_is_user_scoped_and_revisioned() -> None:
    store = InMemorySavedTripStore()
    created = store.create(
        user_id="user-a",
        source_session_id="session-1",
        plan=_plan(),
        title="Quy Nhơn cuối tuần",
    )

    assert created["revision"] == 1
    assert store.get("plan-qn-1", user_id="user-b") is None

    updated = store.update(
        "plan-qn-1",
        user_id="user-a",
        expected_revision=1,
        title="Quy Nhơn 1 ngày",
    )

    assert updated is not None
    assert updated["revision"] == 2
    assert updated["title"] == "Quy Nhơn 1 ngày"

    with pytest.raises(SavedTripRevisionConflictError):
        store.update(
            "plan-qn-1",
            user_id="user-a",
            expected_revision=1,
            title="Stale update",
        )


def test_saved_trip_api_saves_active_plan_and_supports_crud() -> None:
    app = create_app()
    headers = {"X-Dev-User-ID": "trip-owner"}
    with TestClient(app) as client:
        app.state.chat_store.compare_and_set_active_trip_plan(
            "session-1",
            _plan(),
            expected_revision=None,
            user_id="trip-owner",
        )

        created = client.post(
            "/api/trips",
            headers=headers,
            json={"source_session_id": "session-1", "title": "Quy Nhơn của tôi"},
        )
        assert created.status_code == 201
        assert created.json()["trip_id"] == "plan-qn-1"
        assert created.json()["title"] == "Quy Nhơn của tôi"

        listed = client.get("/api/trips", headers=headers)
        assert listed.status_code == 200
        assert len(listed.json()["trips"]) == 1

        changed_plan = created.json()["plan"]
        changed_plan["itinerary"][0]["slots"][0]["start_time"] = "09:00"
        updated = client.patch(
            "/api/trips/plan-qn-1",
            headers=headers,
            json={
                "expected_revision": 1,
                "title": "Quy Nhơn chỉnh sửa",
                "plan": changed_plan,
            },
        )
        assert updated.status_code == 200
        assert updated.json()["revision"] == 2
        assert (
            updated.json()["plan"]["itinerary"][0]["slots"][0]["start_time"]
            == "09:00"
        )

        stale = client.patch(
            "/api/trips/plan-qn-1",
            headers=headers,
            json={"expected_revision": 1, "title": "Bản cũ"},
        )
        assert stale.status_code == 409

        invalid_plan = updated.json()["plan"]
        invalid_plan["itinerary"][0]["slots"][0]["end_time"] = "08:30"
        invalid = client.patch(
            "/api/trips/plan-qn-1",
            headers=headers,
            json={
                "expected_revision": 2,
                "plan": invalid_plan,
            },
        )
        assert invalid.status_code == 422

        deleted = client.delete("/api/trips/plan-qn-1", headers=headers)
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert client.get("/api/trips/plan-qn-1", headers=headers).status_code == 404
