from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from src.app import create_app
from src.config import Settings
from src.security.auth import Principal


def test_principal_normalizes_roles_and_detects_anonymous_users() -> None:
    assert Principal("alice", {"role": "ADMIN"}, "firebase").role == "admin"
    assert Principal("alice", {"role": "owner"}, "firebase").role == "user"
    assert Principal(
        "guest",
        {"firebase": {"sign_in_provider": "anonymous"}},
        "firebase",
    ).is_anonymous
    assert not Principal(
        "alice",
        {"firebase": {"sign_in_provider": "google.com"}},
        "firebase",
    ).is_anonymous
    assert Principal("local-user", {}, "disabled").is_anonymous


def test_disabled_auth_headers_cannot_elevate_server_role() -> None:
    with TestClient(create_app()) as client:
        response = client.get(
            "/api/me",
            headers={"X-Dev-User-ID": "operator", "X-Dev-Role": "support"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "operator",
        "role": "user",
        "auth_mode": "disabled",
        "is_anonymous": True,
    }


def test_production_rejects_disabled_authentication() -> None:
    with pytest.raises(ValidationError, match="AUTH_MODE=firebase"):
        Settings(
            _env_file=None,
            environment="production",
            auth_mode="disabled",
            gemini_context_model="context",
            gemini_answer_model="answer",
            gemini_thinking_level="minimal",
        )
