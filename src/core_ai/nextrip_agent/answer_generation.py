from __future__ import annotations

import json
from typing import Any, Protocol


_REDUNDANT_FACT_GROUPS = (("address", "location"),)


def fact_value_text(value: object) -> str:
    """Render a typed fact consistently for templates and LLM references."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def facts_for_answer(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project factual payloads for prose while preserving the API response facts."""
    selected = list(facts)
    for predicates in _REDUNDANT_FACT_GROUPS:
        preferred, *redundant = predicates
        subjects_with_preferred = {
            str(fact.get("subject_id"))
            for fact in selected
            if fact.get("predicate") == preferred
        }
        selected = [
            fact
            for fact in selected
            if not (
                fact.get("predicate") in redundant
                and str(fact.get("subject_id")) in subjects_with_preferred
            )
        ]
    return selected


class SupportsAnswerGeneration(Protocol):
    def generate(
        self,
        *,
        question: str,
        answer_type: str,
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        matched_paths: list[dict[str, Any]],
        conversation_context: dict[str, Any] | None = None,
    ) -> str: ...


class SupportsAnswerSynthesis(SupportsAnswerGeneration, Protocol):
    def synthesize(
        self,
        *,
        question: str,
        answer_type: str,
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        matched_paths: list[dict[str, Any]],
        weather: dict[str, Any] | None,
        conversation_context: dict[str, Any] | None = None,
        itinerary: list[dict[str, Any]] | None = None,
    ) -> str: ...
