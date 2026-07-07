from __future__ import annotations

from typing import Any, TypedDict


class NexTripAgentState(TypedDict, total=False):
    message: str
    session_id: str
    city: str | None
    entity_types: list[str] | None
    top_k: int
    retrieval_plan: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer: str
    trace: list[dict[str, Any]]
