from __future__ import annotations

from typing import Any

from .models import PersonalizationProfile


def compile_personalization_context(
    profile: PersonalizationProfile | None,
) -> dict[str, Any]:
    """Compile persisted preferences into a bounded, non-sensitive KB contract."""
    if profile is None or not profile.personalization_enabled:
        return {}
    hard_constraints: dict[str, Any] = {}
    if profile.has_children is not None:
        hard_constraints["has_children"] = profile.has_children
    return {
        "profile_revision": profile.profile_revision,
        "budget_level": profile.budget_level,
        "travel_pace": profile.travel_pace,
        "party_type": profile.party_type,
        "hard_constraints": hard_constraints,
        "preferred_concepts": profile.preferred_concepts,
        "excluded_concepts": profile.excluded_concepts,
        "preferred_cities": profile.preferred_cities,
        "dietary_requirements": profile.dietary_requirements,
        "accessibility_requirements": profile.accessibility_requirements,
        "transport_preferences": profile.transport_preferences,
    }

