from __future__ import annotations

from fastapi import APIRouter, Depends

from src.apis.domains.auth.schemas import AuthProfileResponse
from src.security.auth import Principal, current_principal


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/me", response_model=AuthProfileResponse)
async def auth_profile(
    principal: Principal = Depends(current_principal),
) -> AuthProfileResponse:
    permissions = ["chat"]
    if principal.is_admin:
        permissions.extend(["evaluate", "select_kb_version"])
    return AuthProfileResponse(
        uid=principal.uid,
        email=principal.email,
        display_name=principal.display_name,
        role=principal.role,
        auth_mode=principal.auth_mode,
        permissions=permissions,
    )
