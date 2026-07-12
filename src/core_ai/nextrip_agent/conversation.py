from __future__ import annotations

from dataclasses import dataclass
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

    def trace_event(self) -> dict[str, Any]:
        return {
            "node": "context_resolver",
            "status": "completed",
            "city": self.city,
            "city_source": self.city_source,
            "history_messages": self.history_messages,
        }


def resolve_conversation_context(
    *,
    message: str,
    explicit_city: str | None,
    history: list[dict[str, Any]],
) -> ConversationContext:
    city = _canonical_city(explicit_city)
    if city:
        return ConversationContext(city, "request", len(history))

    city = _city_in_text(message)
    if city:
        return ConversationContext(city, "current_message", len(history))

    for item in reversed(history):
        city = _canonical_city(item.get("city"))
        if city:
            return ConversationContext(city, "session_metadata", len(history))
        if item.get("role") == "user":
            city = _city_in_text(str(item.get("content") or ""))
            if city:
                return ConversationContext(city, "conversation_history", len(history))

    return ConversationContext(None, None, len(history))


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
