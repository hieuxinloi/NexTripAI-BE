from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from src.apis.domains.trips.schemas import (
    SavedTrip,
    SavedTripCreateRequest,
    SavedTripDeleteResponse,
    SavedTripListResponse,
    SavedTripUpdateRequest,
)
from src.core_ai.nextrip_agent.trip_plan import ActiveTripPlan
from src.infra.chat_store import ChatStore
from src.infra.saved_trip_store import (
    SavedTripRevisionConflictError,
    SavedTripStore,
)
from src.security.auth import Principal, current_principal


router = APIRouter(prefix="/api/trips", tags=["saved-trips"])


def _store(request: Request) -> SavedTripStore:
    return request.app.state.saved_trip_store


@router.post("", response_model=SavedTrip, status_code=status.HTTP_201_CREATED)
def save_trip(
    payload: SavedTripCreateRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> SavedTrip:
    chat_store: ChatStore = request.app.state.chat_store
    try:
        raw_plan = chat_store.get_active_trip_plan(
            payload.source_session_id,
            user_id=principal.uid,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Session access denied.") from exc
    if raw_plan is None:
        raise HTTPException(status_code=404, detail="The session has no active trip plan.")
    plan = ActiveTripPlan.model_validate(raw_plan)
    if not plan.itinerary:
        raise HTTPException(
            status_code=409,
            detail="The active trip plan has no itinerary to save.",
        )
    title = (payload.title or "").strip() or _default_title(plan)
    return SavedTrip.model_validate(
        _store(request).create(
            user_id=principal.uid,
            source_session_id=payload.source_session_id,
            plan=plan.model_dump(mode="json"),
            title=title,
        )
    )


@router.get("", response_model=SavedTripListResponse)
def list_saved_trips(
    request: Request,
    principal: Principal = Depends(current_principal),
    limit: int = Query(default=50, ge=1, le=100),
) -> SavedTripListResponse:
    return SavedTripListResponse(
        trips=[
            SavedTrip.model_validate(item)
            for item in _store(request).list(user_id=principal.uid, limit=limit)
        ]
    )


@router.get("/{trip_id}", response_model=SavedTrip)
def get_saved_trip(
    request: Request,
    principal: Principal = Depends(current_principal),
    trip_id: str = Path(min_length=1, max_length=128),
) -> SavedTrip:
    item = _store(request).get(trip_id, user_id=principal.uid)
    if item is None:
        raise HTTPException(status_code=404, detail="Saved trip not found.")
    return SavedTrip.model_validate(item)


@router.patch("/{trip_id}", response_model=SavedTrip)
def update_saved_trip(
    payload: SavedTripUpdateRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
    trip_id: str = Path(min_length=1, max_length=128),
) -> SavedTrip:
    title = payload.title.strip() if payload.title is not None else None
    if payload.title is not None and not title:
        raise HTTPException(status_code=422, detail="Saved trip title cannot be empty.")
    if payload.plan is not None and not payload.plan.itinerary:
        raise HTTPException(
            status_code=422,
            detail="A saved trip must contain at least one itinerary day.",
        )
    if payload.plan is not None:
        _validate_itinerary(payload.plan)
    try:
        updated = _store(request).update(
            trip_id,
            user_id=principal.uid,
            expected_revision=payload.expected_revision,
            title=title,
            plan=payload.plan.model_dump(mode="json") if payload.plan else None,
        )
    except SavedTripRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Saved trip not found.")
    return SavedTrip.model_validate(updated)


@router.delete("/{trip_id}", response_model=SavedTripDeleteResponse)
def delete_saved_trip(
    request: Request,
    principal: Principal = Depends(current_principal),
    trip_id: str = Path(min_length=1, max_length=128),
) -> SavedTripDeleteResponse:
    return SavedTripDeleteResponse(
        trip_id=trip_id,
        deleted=_store(request).delete(trip_id, user_id=principal.uid),
    )


def _default_title(plan: ActiveTripPlan) -> str:
    suffix = f"{plan.duration_days} ngày" if plan.duration_days > 1 else "1 ngày"
    return f"Lịch trình {plan.city} · {suffix}"


def _validate_itinerary(plan: ActiveTripPlan) -> None:
    time_pattern = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
    for day in plan.itinerary:
        if not isinstance(day, dict):
            raise HTTPException(status_code=422, detail="Invalid itinerary day.")
        slots = day.get("slots")
        if not isinstance(slots, list):
            raise HTTPException(status_code=422, detail="Invalid itinerary slots.")
        for slot in slots:
            if not isinstance(slot, dict):
                raise HTTPException(status_code=422, detail="Invalid itinerary slot.")
            start_time = str(slot.get("start_time") or "")
            end_time = str(slot.get("end_time") or "")
            if (
                not time_pattern.fullmatch(start_time)
                or not time_pattern.fullmatch(end_time)
                or start_time >= end_time
            ):
                raise HTTPException(
                    status_code=422,
                    detail="Each itinerary stop requires a valid start and end time.",
                )
