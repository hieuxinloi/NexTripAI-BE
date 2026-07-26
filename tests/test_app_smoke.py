from __future__ import annotations

from fastapi.testclient import TestClient

from src.app import create_app


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


def test_app_exposes_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health", headers={"X-Request-ID": "test-request-id"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["worker_pool"]["workers"] == 5
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_kb_versions_only_exposes_ready_versions() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.kb_client = VersionDiscoveryKbClient()
        response = client.get("/api/kb/versions")

    assert response.status_code == 200
    assert response.json()["versions"] == [
        {"kb_version": "v5", "label": "V5"},
        {"kb_version": "v3", "label": "V3"},
    ]


def test_explicit_unavailable_kb_version_is_not_silently_replaced() -> None:
    app = create_app()
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
