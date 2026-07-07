from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KbSearchPayload(BaseModel):
    results: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class AgentResult(BaseModel):
    answer: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
