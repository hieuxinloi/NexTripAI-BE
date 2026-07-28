from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.core_ai.personalization.models import UserAdminUpdate
from src.infra.kb_client import KbClient
from src.infra.user_profile_store import UserProfileStore
from src.security.auth import Principal
from src.security.permissions import require_admin, require_support_or_admin


router = APIRouter(prefix="/api/admin", tags=["admin"])


def _profiles(request: Request) -> UserProfileStore:
    return request.app.state.user_profile_store


@router.get("/users")
def list_users(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    principal: Principal = Depends(require_support_or_admin),
) -> dict[str, Any]:
    profiles = _profiles(request).list_users(limit)
    if principal.auth_mode != "firebase":
        return {"users": profiles}
    return {"users": _merge_firebase_users(profiles, limit)}


@router.get("/users/{user_id}")
def get_user(
    user_id: str,
    request: Request,
    _: Principal = Depends(require_support_or_admin),
) -> dict[str, Any]:
    return _profiles(request).get_profile(user_id).model_dump(mode="json")


@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    payload: UserAdminUpdate,
    request: Request,
    principal: Principal = Depends(require_admin),
) -> dict[str, Any]:
    if user_id == principal.uid and payload.role is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Administrators cannot change their own role.",
        )
    if principal.auth_mode == "firebase":
        _update_firebase_user(user_id, payload)
    result = _profiles(request).update_admin_metadata(
        user_id,
        status=payload.status,
        role=payload.role,
    )
    _profiles(request).record_audit(
        actor_uid=principal.uid,
        action="user.update",
        target_uid=user_id,
        details=payload.model_dump(exclude_none=True),
    )
    return result


@router.post("/users/{user_id}/reset-preferences")
def admin_reset_preferences(
    user_id: str,
    request: Request,
    principal: Principal = Depends(require_admin),
) -> dict[str, Any]:
    profile = _profiles(request).reset_profile(user_id)
    _profiles(request).record_audit(
        actor_uid=principal.uid,
        action="user.preferences.reset",
        target_uid=user_id,
    )
    return profile.model_dump(mode="json")


@router.get("/audit-logs")
def audit_logs(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    _: Principal = Depends(require_admin),
) -> dict[str, Any]:
    return {"events": _profiles(request).list_audit(limit)}


@router.get("/graphrag/deployments")
def graphrag_deployments(
    request: Request,
    _: Principal = Depends(require_support_or_admin),
) -> dict[str, Any]:
    client: KbClient = request.app.state.kb_client
    return client.admin_deployments()


@router.post("/graphrag/deployments/{version}/validate")
def validate_graphrag_deployment(
    version: str,
    request: Request,
    _: Principal = Depends(require_admin),
) -> dict[str, Any]:
    client: KbClient = request.app.state.kb_client
    return client.admin_validate_deployment(version)


@router.post("/graphrag/deployments/{version}/activate")
def activate_graphrag_deployment(
    version: str,
    request: Request,
    principal: Principal = Depends(require_admin),
) -> dict[str, Any]:
    client: KbClient = request.app.state.kb_client
    result = client.admin_activate_deployment(version)
    _profiles(request).record_audit(
        actor_uid=principal.uid,
        action="graphrag.deployment.activate",
        target_uid=version,
        details={"active_version": result.get("active_version")},
    )
    return result


@router.post("/graphrag/rollback")
def rollback_graphrag_deployment(
    request: Request,
    principal: Principal = Depends(require_admin),
) -> dict[str, Any]:
    client: KbClient = request.app.state.kb_client
    result = client.admin_rollback_deployment()
    _profiles(request).record_audit(
        actor_uid=principal.uid,
        action="graphrag.deployment.rollback",
        target_uid=str(result.get("active_version") or ""),
    )
    return result


def _update_firebase_user(user_id: str, payload: UserAdminUpdate) -> None:
    from firebase_admin import auth

    if payload.status is not None:
        auth.update_user(user_id, disabled=payload.status == "suspended")
    if payload.role is not None:
        user = auth.get_user(user_id)
        claims = dict(user.custom_claims or {})
        claims["role"] = payload.role
        auth.set_custom_user_claims(user_id, claims)


def _merge_firebase_users(
    profiles: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    from firebase_admin import auth

    profiles_by_id = {
        str(profile.get("user_id")): profile
        for profile in profiles
        if profile.get("user_id")
    }
    users: list[dict[str, Any]] = []
    page = auth.list_users(max_results=min(limit, 1_000))
    for account in page.iterate_all():
        profile = profiles_by_id.pop(account.uid, {})
        claims = account.custom_claims or {}
        users.append(
            {
                **profile,
                "user_id": account.uid,
                "email": account.email,
                "display_name": account.display_name,
                "status": "suspended"
                if account.disabled
                else profile.get("status", "active"),
                "role_display": claims.get(
                    "role",
                    profile.get("role_display", "user"),
                ),
            }
        )
        if len(users) >= limit:
            return users
    users.extend(list(profiles_by_id.values())[: limit - len(users)])
    return users
