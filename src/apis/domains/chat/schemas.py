from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.core_ai.nextrip_agent.constants import KbVersion
from src.core_ai.nextrip_agent.weather import WeatherAssessment


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    display_message: str | None = Field(default=None, min_length=1, max_length=4000)
    session_id: str = Field(..., min_length=1, max_length=128)
    city: str | None = Field(default=None, max_length=80)
    entity_types: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    kb_version: KbVersion = Field(default="v8", pattern=r"^v[1-9][0-9]*$")
    travel_date: date | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    include_weather: bool | None = None

    @model_validator(mode="after")
    def validate_coordinates(self) -> "ChatRequest":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        return self


class AuthenticatedChatRequest(ChatRequest):
    user_id: str = Field(..., min_length=1, max_length=128)


class EvidenceItem(BaseModel):
    place_id: str
    name: str | None = None
    city: str | None = None
    entity_type: str | None = None
    category: str | None = None
    score: float | None = None
    distance_km: float | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    graph_context: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ClarificationOption(BaseModel):
    label: str
    value: str
    description: str | None = None


class Clarification(BaseModel):
    prompt: str
    field: str
    options: list[ClarificationOption] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    intent: str = "kb_retrieval"
    orchestration_mode: str = "graph_only"
    resolved_context: dict[str, Any] = Field(default_factory=dict)
    kb_version: KbVersion = Field(default="v8", pattern=r"^v[1-9][0-9]*$")
    facts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommendations: list[EvidenceItem] = Field(default_factory=list)
    itinerary: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    query_plan: dict[str, Any] = Field(default_factory=dict)
    matched_paths: list[dict[str, Any]] = Field(default_factory=list)
    constraint_results: list[dict[str, Any]] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    clarification: Clarification | None = None
    weather: WeatherAssessment | None = None
    weather_forecast: list[WeatherAssessment] = Field(default_factory=list)


class ChatMessage(BaseModel):
    message_id: str
    role: str
    content: str
    city: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Any | None = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class SessionUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: Any | None = None
    updated_at: Any | None = None
    message_count: int = 0


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)


class SessionCreateResponse(SessionSummary):
    pass
