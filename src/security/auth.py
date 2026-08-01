from __future__ import annotations

from dataclasses import dataclass

from anyio import to_thread
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import Settings


@dataclass(frozen=True, slots=True)
class Principal:
    uid: str
    claims: dict[str, object]
    auth_mode: str
    email: str | None = None
    display_name: str | None = None
    role: str = ""

    def __post_init__(self) -> None:
        requested_role = self.role or str(self.claims.get("role") or "user")
        normalized_role = requested_role.strip().lower()
        if normalized_role not in {"user", "support", "admin"}:
            normalized_role = "user"
        object.__setattr__(self, "role", normalized_role)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_anonymous(self) -> bool:
        if self.auth_mode != "firebase":
            return True
        firebase_claim = self.claims.get("firebase")
        return (
            isinstance(firebase_claim, dict)
            and firebase_claim.get("sign_in_provider") == "anonymous"
        )


class Authenticator:
    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._bearer = HTTPBearer(auto_error=False)
        self._firebase_auth = None
        if (
            app_settings.environment == "production"
            and app_settings.auth_mode != "firebase"
        ):
            raise RuntimeError(
                "AUTH_MODE=firebase is required when ENVIRONMENT=production."
            )
        if app_settings.auth_mode == "firebase":
            try:
                import firebase_admin
                from firebase_admin import auth
            except ImportError as exc:
                raise RuntimeError(
                    "AUTH_MODE=firebase requires the firebase-admin package."
                ) from exc
            try:
                firebase_admin.get_app()
            except ValueError:
                credential = None
                if app_settings.firestore_credentials_path:
                    from firebase_admin import credentials

                    credential = credentials.Certificate(
                        app_settings.firestore_credentials_path
                    )
                firebase_admin.initialize_app(
                    credential,
                    options={"projectId": app_settings.firebase_project_id}
                    if app_settings.firebase_project_id
                    else None,
                )
            self._firebase_auth = auth

    async def authenticate(self, request: Request) -> Principal:
        if self._settings.auth_mode == "disabled":
            uid = request.headers.get("X-Dev-User-ID", "local-user").strip()
            return Principal(
                uid=uid[:128] or "local-user",
                claims={"role": "user"},
                auth_mode="disabled",
                email=request.headers.get("X-Dev-User-Email") or None,
                display_name=request.headers.get("X-Dev-User-Name") or None,
                role="user",
            )

        credentials: HTTPAuthorizationCredentials | None = await self._bearer(request)
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A Firebase ID token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            claims = await to_thread.run_sync(
                self._verify_token,
                credentials.credentials,
                abandon_on_cancel=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="The Firebase ID token is invalid or expired.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        uid = str(claims.get("uid") or claims.get("sub") or "")
        if not uid:
            raise HTTPException(status_code=401, detail="The token has no user identity.")
        email = _optional_claim(claims.get("email"))
        display_name = _optional_claim(claims.get("name"))
        return Principal(
            uid=uid,
            claims=dict(claims),
            auth_mode="firebase",
            email=email,
            display_name=display_name,
            role=_principal_role(claims),
        )

    def _verify_token(self, token: str) -> dict[str, object]:
        if self._firebase_auth is None:
            raise RuntimeError("Firebase authentication is not initialized")
        return self._firebase_auth.verify_id_token(
            token,
            check_revoked=True,
        )


async def current_principal(request: Request) -> Principal:
    authenticator: Authenticator = request.app.state.authenticator
    return await authenticator.authenticate(request)


async def require_admin(
    principal: Principal = Depends(current_principal),
) -> Principal:
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required.",
        )
    return principal


def _optional_claim(value: object) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None


def _principal_role(
    claims: dict[str, object],
) -> str:
    if claims.get("admin") is True:
        return "admin"
    claimed_role = str(claims.get("role") or "").strip().lower()
    if claimed_role == "admin":
        return "admin"
    roles = claims.get("roles")
    if isinstance(roles, (list, tuple, set)) and any(
        str(item).strip().lower() == "admin" for item in roles
    ):
        return "admin"
    if claimed_role == "support":
        return "support"
    if isinstance(roles, (list, tuple, set)) and any(
        str(item).strip().lower() == "support" for item in roles
    ):
        return "support"
    return "user"
