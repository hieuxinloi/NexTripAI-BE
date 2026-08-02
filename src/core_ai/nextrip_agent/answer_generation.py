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


def fact_display_text(value: object, unit: object | None = None) -> str:
    """Render a fact value without attaching a currency to non-monetary markers.

    Data sources use ``free`` as a sentinel for complimentary access/parking.
    It is a semantic value, not a numeric amount, so currency units must be
    omitted and the Vietnamese UI should present it as ``Miễn phí``.
    """
    rendered = fact_value_text(value).strip()
    if rendered.casefold() in {"free", "complimentary", "miễn phí", "mien phi"}:
        return "Miễn phí"
    unit_text = str(unit or "").strip()
    return f"{rendered} {unit_text}".strip()


def facts_for_answer(
    facts: list[dict[str, Any]],
    *,
    evidence: list[dict[str, Any]] | None = None,
    query_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
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
    requested_fields = set((query_plan or {}).get("requested_fields") or [])
    coordinates_explicitly_requested = (
        "location" in requested_fields and "address" not in requested_fields
    )
    if not coordinates_explicitly_requested:
        subjects_with_city = {
            str(item.get("place_id"))
            for item in evidence or []
            if item.get("place_id") and item.get("city")
        }
        selected = [
            fact
            for fact in selected
            if not (
                fact.get("predicate") == "location"
                and str(fact.get("subject_id")) in subjects_with_city
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
