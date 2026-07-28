from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.core_ai.personalization.models import (
    PersonalizationProfile,
    PersonalizationUpdate,
    PreferenceEvent,
)
from src.infra.user_profile_store import (
    ProfileRevisionConflictError,
    UserProfileStore,
)
from src.security.auth import Principal, current_principal


router = APIRouter(prefix="/api/me", tags=["personalization"])


def _store(request: Request) -> UserProfileStore:
    return request.app.state.user_profile_store


@router.get("/preferences", response_model=PersonalizationProfile)
def get_preferences(
    request: Request,
    principal: Principal = Depends(current_principal),
) -> PersonalizationProfile:
    return _store(request).get_profile(principal.uid)


@router.patch("/preferences", response_model=PersonalizationProfile)
def update_preferences(
    payload: PersonalizationUpdate,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> PersonalizationProfile:
    try:
        return _store(request).update_profile(principal.uid, payload)
    except ProfileRevisionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.delete("/preferences", response_model=PersonalizationProfile)
def reset_preferences(
    request: Request,
    principal: Principal = Depends(current_principal),
) -> PersonalizationProfile:
    return _store(request).reset_profile(principal.uid)


@router.post("/events", status_code=status.HTTP_202_ACCEPTED)
def record_preference_event(
    payload: PreferenceEvent,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> dict[str, bool]:
    _store(request).save_event(principal.uid, payload)
    return {"accepted": True}

