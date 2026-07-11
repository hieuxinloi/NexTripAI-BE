from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KbSearchPayload(BaseModel):
    strategy: str | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class AgentResult(BaseModel):
    answer: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    answer_type: str = "kb_retrieval"
    trace: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
