from __future__ import annotations

from dataclasses import dataclass

from anyio import to_thread
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import Settings


@dataclass(frozen=True, slots=True)
class Principal:
    uid: str
    claims: dict[str, object]
    auth_mode: str

    @property
    def role(self) -> str:
        role = str(self.claims.get("role") or "user").strip().lower()
        return role if role in {"user", "support", "admin"} else "user"

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
            role = request.headers.get("X-Dev-Role", "user").strip().lower()
            return Principal(
                uid=uid[:128] or "local-user",
                claims={"role": role},
                auth_mode="disabled",
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
        return Principal(uid=uid, claims=dict(claims), auth_mode="firebase")

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
