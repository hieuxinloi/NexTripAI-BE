from __future__ import annotations

import pytest

from src.config import settings
from src.security.auth import Authenticator, _principal_role


def test_admin_role_accepts_supported_firebase_custom_claims() -> None:
    assert _principal_role({"admin": True}) == "admin"
    assert _principal_role({"role": "ADMIN"}) == "admin"
    assert _principal_role({"roles": ["user", "admin"]}) == "admin"


def test_support_role_accepts_supported_firebase_custom_claims() -> None:
    assert _principal_role({"role": "SUPPORT"}) == "support"
    assert _principal_role({"roles": ["user", "support"]}) == "support"


def test_verified_email_alone_never_grants_admin() -> None:
    assert _principal_role({
        "email": "admin@example.com",
        "email_verified": True,
    }) == "user"


def test_unknown_claims_remain_normal_user() -> None:
    assert _principal_role({
        "role": "editor",
        "roles": ["reviewer"],
    }) == "user"


def test_production_rejects_disabled_authentication() -> None:
    app_settings = settings().model_copy(
        update={
            "environment": "production",
            "auth_mode": "disabled",
        }
    )
    with pytest.raises(
        RuntimeError,
        match="AUTH_MODE=firebase is required",
    ):
        Authenticator(app_settings)
