from __future__ import annotations

from collections.abc import Iterable

from src.core_ai.personalization.models import (
    PreferenceEventRecord,
    RecommendationFeed,
    SavedPlacesResponse,
)
from src.infra.kb_client import KbClient
from src.infra.user_profile_store import UserProfileStore


TRANSIENT_SEED_EVENTS = frozenset({"ask_place", "view_detail", "open_source"})
STATE_EVENTS: dict[str, tuple[str, bool]] = {
    "like": ("opinion", True),
    "dislike": ("opinion", False),
    "dismiss_recommendation": ("opinion", False),
    "save": ("saved", True),
    "unsave": ("saved", False),
    "add_to_itinerary": ("itinerary", True),
    "remove_from_itinerary": ("itinerary", False),
}


class UnsupportedRecommendationVersionError(ValueError):
    pass


class RecommendationService:
    def __init__(
        self,
        store: UserProfileStore,
        kb_client: KbClient,
        *,
        kb_version: str,
    ) -> None:
        if kb_version != "v8":
            raise UnsupportedRecommendationVersionError(
                "Personalized recommendations currently require Knowledge Base V8."
            )
        self._store = store
        self._kb_client = kb_client
        self._kb_version = kb_version

    def feed(
        self,
        user_id: str,
        *,
        city: str | None,
        seed_place_ids: list[str],
        limit: int,
    ) -> RecommendationFeed:
        profile = self._store.get_profile(user_id)
        events = (
            self._store.recent_events(user_id)
            if profile.personalization_enabled
            else []
        )
        positive_ids, excluded_ids = _current_taste(events)
        saved_ids = self._store.saved_place_ids(user_id, limit=20)
        stored_seeds = (
            [*saved_ids, *positive_ids] if profile.personalization_enabled else []
        )
        seeds = _unique([*seed_place_ids, *stored_seeds])[:20]
        if city:
            cities = [city]
        elif profile.personalization_enabled:
            cities = profile.preferred_cities
        else:
            cities = []
        preferred_concepts = (
            profile.preferred_concepts if profile.personalization_enabled else []
        )
        excluded_concepts = (
            profile.excluded_concepts if profile.personalization_enabled else []
        )
        payload = self._kb_client.personalized_recommendations(
            seed_place_ids=seeds,
            preferred_concepts=preferred_concepts,
            excluded_concepts=excluded_concepts,
            excluded_place_ids=_unique([*excluded_ids, *seeds]),
            preferred_cities=cities,
            limit=limit,
            kb_version=self._kb_version,
        )
        items = [
            item.model_copy(update={"saved": item.place_id in saved_ids})
            for item in payload.items
        ]
        personalized = profile.personalization_enabled and bool(
            seeds or preferred_concepts or cities
        )
        source = "popular"
        if not profile.personalization_enabled:
            source = "disabled"
        elif personalized:
            source = "personalized"
        return RecommendationFeed(
            items=items,
            personalized=personalized,
            source=source,
        )

    def saved_places(self, user_id: str) -> SavedPlacesResponse:
        saved_ids = self._store.saved_place_ids(user_id)
        if not saved_ids:
            return SavedPlacesResponse()
        payload_items = []
        for batch in _chunks(saved_ids, 100):
            payload = self._kb_client.places_by_ids(
                batch,
                kb_version=self._kb_version,
            )
            payload_items.extend(payload.items)
        items = [
            item.model_copy(
                update={
                    "saved": True,
                    "reason_code": "popular",
                    "reason": "Địa điểm bạn đã lưu",
                }
            )
            for item in payload_items
        ]
        return SavedPlacesResponse(items=items)


def _current_taste(
    events: list[PreferenceEventRecord],
) -> tuple[list[str], list[str]]:
    state: dict[tuple[str, str], bool] = {}
    transient_seeds: list[str] = []
    # Stores return newest events first. setdefault keeps the newest transition
    # for each state family while preserving recent transient seed order.
    for event in events:
        if event.event_type in TRANSIENT_SEED_EVENTS:
            transient_seeds.append(event.place_id)
            continue
        transition = STATE_EVENTS.get(event.event_type)
        if transition is not None:
            family, enabled = transition
            state.setdefault((event.place_id, family), enabled)

    excluded = _unique(
        place_id
        for (place_id, family), enabled in state.items()
        if family == "opinion" and not enabled
    )
    excluded_set = set(excluded)

    positive = _unique(
        [
            *(place_id for place_id in transient_seeds if place_id not in excluded_set),
            *(
                place_id
                for (place_id, _), enabled in state.items()
                if enabled and place_id not in excluded_set
            ),
        ]
    )
    return positive, excluded


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
