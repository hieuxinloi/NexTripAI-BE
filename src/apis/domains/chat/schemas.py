from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.core_ai.nextrip_agent.constants import DEFAULT_KB_VERSION, KbVersion


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    city: str | None = None
    entity_types: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    kb_version: KbVersion = DEFAULT_KB_VERSION


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


class ChatResponse(BaseModel):
    answer: str
    intent: str = "kb_retrieval"
    kb_version: KbVersion = DEFAULT_KB_VERSION
    facts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommendations: list[EvidenceItem] = Field(default_factory=list)
    itinerary: list[dict[str, Any]] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    query_plan: dict[str, Any] = Field(default_factory=dict)
    matched_paths: list[dict[str, Any]] = Field(default_factory=list)
    constraint_results: list[dict[str, Any]] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
