from __future__ import annotations

from typing import Any, Protocol


_REDUNDANT_FACT_GROUPS = (("address", "location"),)


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
    ) -> str:
        ...
