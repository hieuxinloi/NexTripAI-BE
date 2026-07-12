from __future__ import annotations

from fastapi.testclient import TestClient

from src.app import create_app


class FailingKbClient:
    def search(self, **kwargs):
        raise RuntimeError("KB is down")


def test_app_exposes_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health", headers={"X-Request-ID": "test-request-id"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["worker_pool"]["workers"] == 5
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_chat_returns_retryable_service_error_when_kb_is_down() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.kb_client = FailingKbClient()
        response = client.post(
            "/api/chat",
            json={"message": "Goi y cafe", "session_id": "test-session"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Knowledge Base is temporarily unavailable."}
