from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from google.api_core.exceptions import GoogleAPICallError
from pydantic import BaseModel

from src.core_ai.personalization.models import (
    PersonalizationProfile,
    PersonalizationUpdate,
    PreferenceEvent,
    RecommendationFeed,
    SavedPlacesResponse,
)
from src.core_ai.personalization.recommendations import (
    RecommendationService,
    UnsupportedRecommendationVersionError,
)
from src.infra.user_profile_store import (
    ProfileRevisionConflictError,
    UserProfileStore,
)
from src.infra.resilience import CircuitOpenError
from src.security.auth import Principal, current_principal


router = APIRouter(prefix="/api/me", tags=["personalization"])


class SessionIdentity(BaseModel):
    user_id: str
    role: str
    auth_mode: str
    is_anonymous: bool


@router.get("", response_model=SessionIdentity)
def get_session_identity(
    principal: Principal = Depends(current_principal),
) -> SessionIdentity:
    return SessionIdentity(
        user_id=principal.uid,
        role=principal.role,
        auth_mode=principal.auth_mode,
        is_anonymous=principal.is_anonymous,
    )


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


@router.get("/recommendations", response_model=RecommendationFeed)
def get_recommendations(
    request: Request,
    city: str | None = Query(default=None, max_length=80),
    seed_place_ids: list[str] = Query(default_factory=list, max_length=20),
    limit: int = Query(default=6, ge=1, le=12),
    principal: Principal = Depends(current_principal),
) -> RecommendationFeed:
    try:
        return RecommendationService(
            _store(request),
            request.app.state.kb_client,
            kb_version=request.app.state.settings.active_kb_version,
        ).feed(
            principal.uid,
            city=city,
            seed_place_ids=seed_place_ids,
            limit=limit,
        )
    except UnsupportedRecommendationVersionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (
        httpx.HTTPError,
        CircuitOpenError,
        GoogleAPICallError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Personalized recommendations are temporarily unavailable.",
        ) from exc


@router.get("/saved-places", response_model=SavedPlacesResponse)
def get_saved_places(
    request: Request,
    principal: Principal = Depends(current_principal),
) -> SavedPlacesResponse:
    try:
        return RecommendationService(
            _store(request),
            request.app.state.kb_client,
            kb_version=request.app.state.settings.active_kb_version,
        ).saved_places(principal.uid)
    except UnsupportedRecommendationVersionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (
        httpx.HTTPError,
        CircuitOpenError,
        GoogleAPICallError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Saved places are temporarily unavailable.",
        ) from exc
