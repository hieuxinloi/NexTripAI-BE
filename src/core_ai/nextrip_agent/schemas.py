from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KbSearchPayload(BaseModel):
    strategy: str | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class TypedCitation(BaseModel):
    subject_id: str | None = None
    source_name: str | None = None
    url: str | None = None


class TypedTarget(BaseModel):
    target_id: str
    kind: str
    name: str
    description: str | None = None
    score: float = 0


class TypedServiceError(BaseModel):
    code: str
    message: str
    retryable: bool
    reason: str | None = None


class TypedKbPayload(BaseModel):
    kb_version: str
    answer_type: str = "kb_retrieval"
    entities: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    targets: list[TypedTarget] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[TypedCitation] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    query_plan: dict[str, Any] = Field(default_factory=dict)
    matched_paths: list[dict[str, Any]] = Field(default_factory=list)
    constraint_results: list[dict[str, Any]] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    itinerary: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    conversation_context: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    error: TypedServiceError | None = None


class AgentResult(BaseModel):
    answer: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    answer_type: str = "kb_retrieval"
    trace: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    query_plan: dict[str, Any] = Field(default_factory=dict)
    matched_paths: list[dict[str, Any]] = Field(default_factory=list)
    constraint_results: list[dict[str, Any]] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    itinerary: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    conversation_context: dict[str, Any] = Field(default_factory=dict)
