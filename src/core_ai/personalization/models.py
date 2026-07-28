from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


BudgetLevel = Literal["budget", "moderate", "premium"]
TravelPace = Literal["relaxed", "balanced", "packed"]
PartyType = Literal["solo", "couple", "family", "friends", "business"]
UserStatus = Literal["active", "suspended", "deletion_pending"]


class PersonalizationProfile(BaseModel):
    user_id: str = ""
    personalization_enabled: bool = True
    onboarding_completed: bool = False
    onboarding_version: int = Field(default=0, ge=0)
    profile_revision: int = Field(default=0, ge=0)
    status: UserStatus = "active"
    budget_level: BudgetLevel | None = None
    travel_pace: TravelPace | None = None
    party_type: PartyType | None = None
    has_children: bool | None = None
    preferred_concepts: list[str] = Field(default_factory=list, max_length=30)
    excluded_concepts: list[str] = Field(default_factory=list, max_length=30)
    preferred_cities: list[str] = Field(default_factory=list, max_length=10)
    dietary_requirements: list[str] = Field(default_factory=list, max_length=20)
    accessibility_requirements: list[str] = Field(default_factory=list, max_length=20)
    transport_preferences: list[str] = Field(default_factory=list, max_length=10)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator(
        "preferred_concepts",
        "excluded_concepts",
        "preferred_cities",
        "dietary_requirements",
        "accessibility_requirements",
        "transport_preferences",
    )
    @classmethod
    def normalize_values(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(cleaned))


class PersonalizationUpdate(BaseModel):
    personalization_enabled: bool | None = None
    onboarding_completed: bool | None = None
    onboarding_version: int | None = Field(default=None, ge=0)
    expected_revision: int | None = Field(default=None, ge=0)
    budget_level: BudgetLevel | None = None
    travel_pace: TravelPace | None = None
    party_type: PartyType | None = None
    has_children: bool | None = None
    preferred_concepts: list[str] | None = Field(default=None, max_length=30)
    excluded_concepts: list[str] | None = Field(default=None, max_length=30)
    preferred_cities: list[str] | None = Field(default=None, max_length=10)
    dietary_requirements: list[str] | None = Field(default=None, max_length=20)
    accessibility_requirements: list[str] | None = Field(default=None, max_length=20)
    transport_preferences: list[str] | None = Field(default=None, max_length=10)

    @field_validator(
        "preferred_concepts",
        "excluded_concepts",
        "preferred_cities",
        "dietary_requirements",
        "accessibility_requirements",
        "transport_preferences",
    )
    @classmethod
    def normalize_optional_values(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(cleaned))


class PreferenceEvent(BaseModel):
    event_id: str = Field(min_length=8, max_length=128)
    event_type: Literal[
        "like",
        "dislike",
        "save",
        "unsave",
        "add_to_itinerary",
        "remove_from_itinerary",
        "open_source",
    ]
    place_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = Field(default=None, max_length=128)


class UserAdminUpdate(BaseModel):
    status: UserStatus | None = None
    role: Literal["user", "support", "admin"] | None = None

