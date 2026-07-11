from __future__ import annotations

from typing import Any, TypedDict

from src.core_ai.nextrip_agent.retrieval_plan import RetrievalRequest


class NexTripAgentState(TypedDict, total=False):
    message: str
    session_id: str
    city: str | None
    entity_types: list[str] | None
    top_k: int
    kb_version: str
    retrieval_plan: list[RetrievalRequest]
    evidence: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    missing_fields: list[str]
    answer_type: str
    answer: str
    error: dict[str, Any]
    trace: list[dict[str, Any]]
