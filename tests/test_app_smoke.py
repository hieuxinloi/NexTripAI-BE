from __future__ import annotations

from fastapi.testclient import TestClient

from src.app import create_app


def test_app_exposes_health() -> None:
    client = TestClient(create_app())
    response = client.get("/health", headers={"X-Request-ID": "test-request-id"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"] == "test-request-id"
