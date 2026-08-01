from __future__ import annotations

from fastapi.testclient import TestClient

from src.app import create_app
from src.security.auth import Principal, current_principal


def firebase_admin_principal() -> Principal:
    return Principal(
        uid="firebase-admin-1",
        claims={"role": "admin"},
        auth_mode="firebase",
        email="admin@example.com",
    )


class FailingKbClient:
    def search(self, **kwargs):
        raise RuntimeError("KB is down")


class AvailableTypedKbClient:
    def ready_versions(self):
        return {"v7"}

    def query_typed(self, *, query, top_k, kb_version):
        return {
            "kb_version": kb_version,
            "answer_type": "recommendation",
            "recommendations": [{
                "place_id": "attr_qn_001",
                "name": "Kỳ Co",
                "city": "Quy Nhơn",
                "entity_type": "attraction",
            }],
            "evidence": [],
            "facts": [],
            "missing_fields": [],
            "query_plan": {},
            "matched_paths": [],
            "constraint_results": [],
            "required_tools": [],
            "trace": [],
        }


class VersionDiscoveryKbClient:
    def readiness(self, *, force=False):
        return {
            "status": "ready",
            "ready_versions": ["v3", "v5"],
            "versions": {"v3": "ready", "v5": "ready"},
        }

    def ready_versions(self):
        return {"v3", "v5"}


class OnlyV7ReadyKbClient:
    def ready_versions(self):
        return {"v7"}


class MultiVersionTypedKbClient:
    def __init__(self, response_version: str | None = None) -> None:
        self.response_version = response_version
        self.requested_versions: list[str] = []

    def readiness(self, *, force=False):
        return {
            "status": "ready",
            "active_version": "v8",
            "ready_versions": ["v5", "v8"],
        }

    def query_typed(self, *, query, top_k, kb_version):
        self.requested_versions.append(kb_version)
        return {
            "kb_version": self.response_version or kb_version,
            "answer_type": "recommendation",
            "recommendations": [],
            "evidence": [],
            "facts": [],
            "missing_fields": [],
            "query_plan": {},
            "matched_paths": [],
            "constraint_results": [],
            "required_tools": [],
            "trace": [],
        }


def test_app_exposes_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health", headers={"X-Request-ID": "test-request-id"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["worker_pool"]["workers"] == 5
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_kb_versions_only_exposes_ready_versions() -> None:
    app = create_app()
    app.dependency_overrides[current_principal] = firebase_admin_principal
    with TestClient(app) as client:
        app.state.kb_client = VersionDiscoveryKbClient()
        app.state.settings.allow_client_kb_version = True
        response = client.get("/api/kb/versions")

    assert response.status_code == 200
    assert response.json()["versions"] == [
        {
            "kb_version": "v5",
            "label": "V5",
            "ready": True,
            "selectable": True,
        },
        {
            "kb_version": "v3",
            "label": "V3",
            "ready": True,
            "selectable": True,
        },
    ]
    assert response.json()["active_version"] == "v5"
    assert response.json()["selection_enabled"] is True


def test_auth_profile_exposes_role_permissions() -> None:
    app = create_app()
    with TestClient(app) as client:
        user = client.get(
            "/api/auth/me",
            headers={
                "X-Dev-User-ID": "traveler-1",
                "X-Dev-User-Role": "user",
                "X-Dev-User-Email": "traveler@example.com",
            },
        )
        app.dependency_overrides[current_principal] = firebase_admin_principal
        admin = client.get("/api/auth/me")

    assert user.status_code == 200
    assert user.json() == {
        "uid": "traveler-1",
        "email": "traveler@example.com",
        "display_name": None,
        "role": "user",
        "auth_mode": "disabled",
        "permissions": ["chat"],
    }
    assert admin.json()["role"] == "admin"
    assert admin.json()["auth_mode"] == "firebase"
    assert admin.json()["permissions"] == [
        "chat",
        "evaluate",
        "select_kb_version",
    ]


def test_normal_user_cannot_access_admin_features() -> None:
    app = create_app()
    headers = {
        "X-Dev-User-ID": "traveler-1",
        "X-Dev-User-Role": "user",
    }
    with TestClient(app) as client:
        app.state.kb_client = VersionDiscoveryKbClient()
        versions = client.get("/api/kb/versions", headers=headers)
        evaluations = client.get("/api/evaluations", headers=headers)
        selected_chat = client.post(
            "/api/chat",
            headers=headers,
            json={
                "message": "Goi y cafe",
                "session_id": "user-selected-version",
                "kb_version": "v5",
            },
        )

    assert versions.status_code == 403
    assert evaluations.status_code == 403
    assert selected_chat.status_code == 403
    assert selected_chat.json() == {
        "detail": "Only admins can select a Knowledge Base version."
    }


def test_normal_user_chat_uses_active_knowledge_base() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.kb_client = AvailableTypedKbClient()
        app.state.settings.active_kb_version = "v7"
        response = client.post(
            "/api/chat",
            headers={
                "X-Dev-User-ID": "traveler-1",
                "X-Dev-User-Role": "user",
            },
            json={
                "message": "Goi y cafe",
                "session_id": "user-default-version",
            },
        )

    assert response.status_code != 403


def test_explicit_unavailable_kb_version_is_not_silently_replaced() -> None:
    app = create_app()
    app.dependency_overrides[current_principal] = firebase_admin_principal
    with TestClient(app) as client:
        app.state.kb_client = VersionDiscoveryKbClient()
        app.state.settings.allow_client_kb_version = True
        response = client.post(
            "/api/chat",
            json={
                "message": "Goi y cafe",
                "session_id": "strict-version-session",
                "kb_version": "v4",
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "No configured Knowledge Base version is ready: ['v4']"
    }


def test_explicit_kb_version_selection_is_strict() -> None:
    app = create_app()
    app.dependency_overrides[current_principal] = firebase_admin_principal
    kb_client = MultiVersionTypedKbClient()
    with TestClient(app) as client:
        app.state.kb_client = kb_client
        app.state.settings.allow_client_kb_version = True
        response = client.post(
            "/api/chat",
            json={
                "message": "Goi y cafe o Quy Nhon",
                "session_id": "strict-v5-session",
                "kb_version": "v5",
            },
        )

    assert response.status_code == 200
    assert kb_client.requested_versions == ["v5"]
    assert response.json()["kb_version"] == "v5"


def test_explicit_kb_version_is_rejected_when_selection_is_disabled() -> None:
    app = create_app()
    app.dependency_overrides[current_principal] = firebase_admin_principal
    with TestClient(app) as client:
        app.state.kb_client = MultiVersionTypedKbClient()
        app.state.settings.allow_client_kb_version = False
        response = client.post(
            "/api/chat",
            json={
                "message": "Goi y cafe o Quy Nhon",
                "session_id": "disabled-selection-session",
                "kb_version": "v5",
            },
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Knowledge Base version selection is disabled."
    }


def test_kb_version_mismatch_is_blocked() -> None:
    app = create_app()
    app.dependency_overrides[current_principal] = firebase_admin_principal
    kb_client = MultiVersionTypedKbClient(response_version="v8")
    with TestClient(app) as client:
        app.state.kb_client = kb_client
        app.state.settings.allow_client_kb_version = True
        response = client.post(
            "/api/chat",
            json={
                "message": "Goi y cafe o Quy Nhon",
                "session_id": "mismatch-session",
                "kb_version": "v5",
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Knowledge Base version mismatch: requested V5, received V8."
    }


def test_conversation_cannot_switch_knowledge_base_version_midstream() -> None:
    app = create_app()
    app.dependency_overrides[current_principal] = firebase_admin_principal
    kb_client = MultiVersionTypedKbClient()
    with TestClient(app) as client:
        app.state.kb_client = kb_client
        app.state.settings.allow_client_kb_version = True
        first = client.post(
            "/api/chat",
            json={
                "message": "Goi y cafe o Quy Nhon",
                "session_id": "pinned-version-session",
                "kb_version": "v5",
            },
        )
        switched = client.post(
            "/api/chat",
            json={
                "message": "Tim them nha hang",
                "session_id": "pinned-version-session",
                "kb_version": "v8",
            },
        )

    assert first.status_code == 200
    assert switched.status_code == 409
    assert switched.json() == {
        "detail": (
            "This conversation is pinned to Knowledge Base V5. "
            "Start a new conversation to use V8."
        )
    }
    assert kb_client.requested_versions == ["v5"]


def test_chat_returns_retryable_service_error_when_kb_is_down() -> None:
    app = create_app()
    with TestClient(app) as client:
        expected_version = app.state.settings.active_kb_version.upper()
        app.state.kb_client = FailingKbClient()
        response = client.post(
            "/api/chat",
            json={"message": "Goi y cafe", "session_id": "test-session"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": f"Knowledge Base {expected_version} is temporarily unavailable."
    }


def test_stream_emits_structured_error_when_kb_is_down() -> None:
    app = create_app()
    with TestClient(app) as client:
        expected_version = app.state.settings.active_kb_version.upper()
        app.state.kb_client = FailingKbClient()
        response = client.post(
            "/api/chat/stream",
            headers={"Idempotency-Key": "request-12345"},
            json={"message": "Goi y cafe", "session_id": "stream-session"},
        )

    assert response.status_code == 200
    assert "event: accepted" in response.text
    assert "event: error" in response.text
    assert f"Knowledge Base {expected_version} is temporarily unavailable." in response.text


def test_explicit_unavailable_version_does_not_silently_fallback() -> None:
    app = create_app()
    app.dependency_overrides[current_principal] = firebase_admin_principal
    with TestClient(app) as client:
        app.state.kb_client = OnlyV7ReadyKbClient()
        response = client.post(
            "/api/chat",
            json={
                "message": "Goi y cafe",
                "session_id": "explicit-version-session",
                "kb_version": "v2",
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "No configured Knowledge Base version is ready: ['v2']"
    }


def test_chat_does_not_render_graph_fallback_without_gemini() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.kb_client = AvailableTypedKbClient()
        app.state.settings.active_kb_version = "v7"
        app.state.answer_generator = None
        response = client.post(
            "/api/chat",
            json={"message": "Gợi ý điểm tham quan", "session_id": "no-gemini-session"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Gemini answer generation is temporarily unavailable."
    }


def test_session_history_and_delete_are_scoped_to_authenticated_user() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.chat_store.save_message(
            "owned-session",
            "message-1",
            "user",
            "Xin chao",
            user_id="local-user",
        )
        history = client.get("/api/sessions/owned-session/messages")
        deleted = client.delete("/api/sessions/owned-session")
        empty = client.get("/api/sessions/owned-session/messages")

    assert history.status_code == 200
    assert history.json()["messages"][0]["content"] == "Xin chao"
    assert deleted.json()["deleted"] is True
    assert empty.json()["messages"] == []


def test_session_can_be_renamed() -> None:
    app = create_app()
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={"title": "Tên cũ"})
        session_id = created.json()["session_id"]
        renamed = client.patch(
            f"/api/sessions/{session_id}",
            json={"title": "  Lịch trình Quy Nhơn  "},
        )

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Lịch trình Quy Nhơn"
