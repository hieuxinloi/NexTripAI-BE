from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from src.core_ai.nextrip_agent.trip_plan import ActiveTripPlan


class SavedTripCreateRequest(BaseModel):
    source_session_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    title: str | None = Field(default=None, max_length=120)


class SavedTripUpdateRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, max_length=120)
    plan: ActiveTripPlan | None = None

    @model_validator(mode="after")
    def require_change(self) -> "SavedTripUpdateRequest":
        if self.title is None and self.plan is None:
            raise ValueError("At least one saved-trip field must be updated.")
        return self


class SavedTrip(BaseModel):
    trip_id: str
    title: str
    source_session_id: str | None = None
    plan: ActiveTripPlan
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class SavedTripListResponse(BaseModel):
    trips: list[SavedTrip] = Field(default_factory=list)


class SavedTripDeleteResponse(BaseModel):
    trip_id: str
    deleted: bool
