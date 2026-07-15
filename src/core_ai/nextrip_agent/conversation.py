from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from src.core_ai.nextrip_agent.weather import normalize_text


SUPPORTED_CITIES = {
    "da nang": "Đà Nẵng",
    "quy nhon": "Quy Nhơn",
}


@dataclass(frozen=True)
class ConversationContext:
    city: str | None
    city_source: str | None
    history_messages: int
    travel_date: date | None = None
    travel_date_source: str | None = None
    recent_place_ids: tuple[str, ...] = ()

    def trace_event(self) -> dict[str, Any]:
        return {
            "node": "context_resolver",
            "status": "completed",
            "city": self.city,
            "city_source": self.city_source,
            "history_messages": self.history_messages,
            "travel_date": self.travel_date.isoformat() if self.travel_date else None,
            "travel_date_source": self.travel_date_source,
            "recent_place_ids": list(self.recent_place_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "city_source": self.city_source,
            "travel_date": self.travel_date.isoformat() if self.travel_date else None,
            "travel_date_source": self.travel_date_source,
            "recent_place_ids": list(self.recent_place_ids),
        }


def resolve_conversation_context(
    *,
    message: str,
    explicit_city: str | None,
    history: list[dict[str, Any]],
    explicit_travel_date: date | None = None,
) -> ConversationContext:
    history_date, recent_place_ids = _history_state(history)
    travel_date = explicit_travel_date or history_date
    travel_date_source = "request" if explicit_travel_date else ("session_metadata" if history_date else None)
    city = _canonical_city(explicit_city)
    if city:
        return ConversationContext(city, "request", len(history), travel_date, travel_date_source, recent_place_ids)

    city = _city_in_text(message)
    if city:
        return ConversationContext(city, "current_message", len(history), travel_date, travel_date_source, recent_place_ids)

    for item in reversed(history):
        city = _canonical_city(item.get("city"))
        if city:
            return ConversationContext(city, "session_metadata", len(history), travel_date, travel_date_source, recent_place_ids)
        if item.get("role") == "user":
            city = _city_in_text(str(item.get("content") or ""))
            if city:
                return ConversationContext(city, "conversation_history", len(history), travel_date, travel_date_source, recent_place_ids)

    return ConversationContext(None, None, len(history), travel_date, travel_date_source, recent_place_ids)


def _history_state(history: list[dict[str, Any]]) -> tuple[date | None, tuple[str, ...]]:
    travel_date = None
    place_ids: list[str] = []
    for item in reversed(history):
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if travel_date is None:
            raw_date = metadata.get("travel_date")
            if isinstance(raw_date, str):
                try:
                    travel_date = date.fromisoformat(raw_date)
                except ValueError:
                    pass
        for place_id in metadata.get("place_ids") or []:
            value = str(place_id)
            if value not in place_ids:
                place_ids.append(value)
    return travel_date, tuple(place_ids[:20])


def _canonical_city(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = normalize_text(value)
    for key, city in SUPPORTED_CITIES.items():
        if key in normalized:
            return city
    return value.strip()


def _city_in_text(value: str) -> str | None:
    normalized = normalize_text(value)
    for key, city in SUPPORTED_CITIES.items():
        if key in normalized:
            return city
    return None
