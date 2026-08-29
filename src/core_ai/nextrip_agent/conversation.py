from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from src.core_ai.nextrip_agent.trip_plan import PlanMutation, PlanOperation
from src.core_ai.nextrip_agent.weather import normalize_text


SUPPORTED_CITIES = {
    "da nang": "Đà Nẵng",
    "quy nhon": "Quy Nhơn",
}


class ConversationResolution(BaseModel):
    """A model-produced decision for one turn of a multi-turn conversation."""

    route: Literal["travel", "conversation"] = "travel"
    standalone_message: str = Field(min_length=1, max_length=4000)
    direct_answer: str | None = Field(default=None, max_length=4000)
    summary: str | None = Field(default=None, max_length=4000)
    confidence: float = Field(default=1.0, ge=0, le=1)
    plan_mutation: PlanMutation = Field(default_factory=PlanMutation)

    @model_validator(mode="after")
    def conversation_requires_an_answer(self) -> "ConversationResolution":
        if self.route == "conversation" and not (self.direct_answer or "").strip():
            raise ValueError("A conversation route requires direct_answer")
        if self.route == "travel":
            self.direct_answer = None
        if self.plan_mutation.operation is not PlanOperation.NONE:
            self.route = "travel"
            self.direct_answer = None
        return self


class SupportsConversationContextualization(Protocol):
    def contextualize(
        self,
        *,
        message: str,
        history: list[dict[str, Any]],
        prior_summary: str | None,
        structured_context: dict[str, Any],
    ) -> ConversationResolution: ...


@dataclass(frozen=True)
class ConversationContext:
    city: str | None
    city_source: str | None
    history_messages: int
    travel_date: date | None = None
    travel_date_source: str | None = None
    recent_place_ids: tuple[str, ...] = ()
    recent_dishes: tuple[str, ...] = ()

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
            "recent_dishes": list(self.recent_dishes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "city_source": self.city_source,
            "travel_date": self.travel_date.isoformat() if self.travel_date else None,
            "travel_date_source": self.travel_date_source,
            "recent_place_ids": list(self.recent_place_ids),
            "recent_dishes": list(self.recent_dishes),
        }


@dataclass(frozen=True)
class ResolvedTurn:
    resolution: ConversationResolution
    status: Literal["completed", "fallback", "skipped"]
    reason: str
    original_message: str

    def trace_event(self) -> dict[str, Any]:
        return {
            "node": "conversation_contextualizer",
            "status": self.status,
            "route": self.resolution.route,
            "contextualized": (
                self.resolution.standalone_message.strip()
                != self.original_message.strip()
            ),
            "confidence": self.resolution.confidence,
            "reason": self.reason,
        }


def resolve_conversation_context(
    *,
    message: str,
    explicit_city: str | None,
    history: list[dict[str, Any]],
    explicit_travel_date: date | None = None,
) -> ConversationContext:
    history_date, recent_place_ids, recent_dishes = _history_state(history)
    travel_date = explicit_travel_date or history_date
    travel_date_source = (
        "request"
        if explicit_travel_date
        else ("session_metadata" if history_date else None)
    )
    city = _city_in_text(message)
    if city:
        return ConversationContext(
            city,
            "current_message",
            len(history),
            travel_date,
            travel_date_source,
            recent_place_ids,
            recent_dishes,
        )

    city = _canonical_city(explicit_city)
    if city:
        return ConversationContext(
            city,
            "request",
            len(history),
            travel_date,
            travel_date_source,
            recent_place_ids,
            recent_dishes,
        )

    for item in reversed(history):
        city = _canonical_city(item.get("city"))
        if city:
            return ConversationContext(
                city,
                "session_metadata",
                len(history),
                travel_date,
                travel_date_source,
                recent_place_ids,
                recent_dishes,
            )
        if item.get("role") == "user":
            city = _city_in_text(str(item.get("content") or ""))
            if city:
                return ConversationContext(
                    city,
                    "conversation_history",
                    len(history),
                    travel_date,
                    travel_date_source,
                    recent_place_ids,
                    recent_dishes,
                )

    return ConversationContext(
        None,
        None,
        len(history),
        travel_date,
        travel_date_source,
        recent_place_ids,
        recent_dishes,
    )


def resolve_turn(
    *,
    message: str,
    history: list[dict[str, Any]],
    context: ConversationContext,
    contextualizer: SupportsConversationContextualization | None,
    prior_summary: str | None = None,
    active_trip_plan: dict[str, Any] | None = None,
) -> ResolvedTurn:
    """Resolve references generically; never make chat availability depend on Gemini."""
    fallback = ConversationResolution(
        route="travel",
        standalone_message=message,
        summary=prior_summary,
        confidence=0,
    )
    if not history:
        return ResolvedTurn(fallback, "skipped", "first_turn", message)
    if contextualizer is None:
        return ResolvedTurn(
            fallback,
            "skipped",
            "contextualizer_not_configured",
            message,
        )
    try:
        structured_context = context.to_dict()
        if active_trip_plan is not None:
            structured_context["active_trip_plan"] = active_trip_plan
        resolution = contextualizer.contextualize(
            message=message,
            history=history,
            prior_summary=prior_summary,
            structured_context=structured_context,
        )
    except Exception as exc:
        logger.warning(
            "Conversation contextualization failed error_type={}; using original message",
            exc.__class__.__name__,
        )
        return ResolvedTurn(fallback, "fallback", exc.__class__.__name__, message)
    return ResolvedTurn(resolution, "completed", "model_decision", message)


def should_preserve_named_target_message(
    *,
    message: str,
    standalone_message: str,
    context: ConversationContext,
) -> bool:
    """Avoid applying an inherited city to a newly named venue."""

    if context.city_source not in {"session_metadata", "conversation_history"}:
        return False
    if not context.city or _city_in_text(message) is not None:
        return False
    if _city_in_text(standalone_message) != context.city:
        return False
    normalized = normalize_text(message)
    if not any(
        marker in normalized
        for marker in (
            "khach san",
            "hotel",
            "resort",
            "nha hang",
            "quan an",
            "quan cafe",
            "quan ca phe",
            "diem tham quan",
            "khu du lich",
        )
    ):
        return False
    generic = {
        "cho",
        "toi",
        "xin",
        "biet",
        "thong",
        "tin",
        "gia",
        "phong",
        "khach",
        "san",
        "cua",
        "ngay",
        "dem",
        "vao",
        "tai",
        "o",
        "den",
        "tu",
        "goi",
        "y",
        "khong",
        "co",
        "a",
    }
    return any(
        len(token) >= 3 and token not in generic
        for token in normalized.split()
    )


def answer_memory_context(
    *,
    history: list[dict[str, Any]],
    resolved_turn: ResolvedTurn,
) -> dict[str, Any] | None:
    if not history:
        return None
    turns = []
    for item in history:
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            turns.append({"role": role, "content": content[:1600]})
    return {
        "summary": resolved_turn.resolution.summary,
        "recent_turns": turns,
        "standalone_question": resolved_turn.resolution.standalone_message,
    }


def _history_state(
    history: list[dict[str, Any]],
) -> tuple[date | None, tuple[str, ...], tuple[str, ...]]:
    travel_date = None
    place_ids: list[str] = []
    dish_names: list[str] = []
    found_entity_snapshot = False
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
        if not found_entity_snapshot and "referenced_entities" in metadata:
            found_entity_snapshot = True
            for entity in metadata.get("referenced_entities") or []:
                if not isinstance(entity, dict) or entity.get("kind") != "dish":
                    continue
                name = str(entity.get("name") or "").strip()
                if name and name not in dish_names:
                    dish_names.append(name)
    return travel_date, tuple(place_ids[:20]), tuple(dish_names[:10])


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
