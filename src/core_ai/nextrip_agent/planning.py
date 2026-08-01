from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, Field

from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.weather import WeatherAssessment, normalize_text


class ItineraryRole(StrEnum):
    ACTIVITY = "activity"
    MEAL = "meal"
    CAFE_BREAK = "cafe_break"
    CHECK_IN = "check_in"
    REST = "rest"
    CHECK_OUT = "check_out"


class PlannedStop(BaseModel):
    start_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    place_id: str = Field(min_length=1)
    role: ItineraryRole
    rationale: str = Field(min_length=1, max_length=300)


class PlannedDay(BaseModel):
    day: int = Field(ge=1, le=30)
    stops: list[PlannedStop] = Field(min_length=1, max_length=8)


class ItineraryPlan(BaseModel):
    days: list[PlannedDay] = Field(min_length=1, max_length=30)
    summary: str = Field(default="", max_length=500)


class PlannedStopDraft(BaseModel):
    """Gemini-facing schema without unsupported JSON Schema constraints."""

    start_time: str
    end_time: str
    place_id: str
    role: ItineraryRole
    rationale: str


class PlannedDayDraft(BaseModel):
    day: int
    stops: list[PlannedStopDraft]


class ItineraryPlanDraft(BaseModel):
    days: list[PlannedDayDraft]
    summary: str = ""


class SupportsItineraryPlanning(Protocol):
    def plan_itinerary(
        self,
        *,
        question: str,
        city: str,
        duration_days: int,
        duration_nights: int,
        candidates: list[dict[str, Any]],
        weather: list[dict[str, Any]],
        latitude: float | None,
        longitude: float | None,
        personalization_context: dict[str, Any] | None = None,
    ) -> ItineraryPlan: ...


ITINERARY_TERMS = (
    "lich trinh",
    "lo trinh",
    "ke hoach chuyen di",
    "len ke hoach",
    "sap xep chuyen di",
)

PLANNABLE_ENTITY_TYPES = frozenset(
    {"attraction", "restaurant", "cafe", "hotel", "nightlife"}
)


def is_itinerary_request(message: str, query_plan: dict[str, Any] | None = None) -> bool:
    if (query_plan or {}).get("intent") == "plan_candidates":
        return True
    normalized = normalize_text(message)
    return any(term in normalized for term in ITINERARY_TERMS)


def planning_agent_node(
    *,
    message: str,
    graph: AgentResult,
    weather_forecast: list[WeatherAssessment],
    planner: SupportsItineraryPlanning | None,
    city: str | None,
    latitude: float | None,
    longitude: float | None,
    personalization_context: dict[str, Any] | None = None,
) -> tuple[AgentResult, dict[str, Any]]:
    if not is_itinerary_request(message, graph.query_plan):
        return graph, {"node": "planning", "status": "skipped"}

    duration_days = requested_itinerary_duration_days(message, graph.query_plan)
    duration_nights = _duration_nights(message, duration_days)
    resolved_city = city or _single_candidate_city(graph.evidence)
    if resolved_city is None:
        missing = list(dict.fromkeys([*graph.missing_fields, "city"]))
        return graph.model_copy(update={"missing_fields": missing}), {
            "node": "planning",
            "status": "needs_input",
            "missing": "city",
        }

    candidates = _city_candidates(graph.evidence, resolved_city)
    if not candidates:
        return graph, {
            "node": "planning",
            "status": "unavailable",
            "reason": "no_grounded_candidates_in_city",
            "city": resolved_city,
        }

    planning_warnings: list[str] = []
    source = "deterministic_fallback"
    plan: ItineraryPlan | None = None
    planner_candidates = _weather_safe_candidates(candidates, weather_forecast)
    safe_activity_count = sum(
        item.get("entity_type") in {"attraction", "nightlife"}
        for item in planner_candidates
    )
    if safe_activity_count < duration_days:
        planning_warnings.append("limited_weather_safe_activities")
    if not _has_fallback_candidates(
        planner_candidates,
        duration_nights=duration_nights,
    ):
        planning_warnings.append("no_plannable_candidates")
        updated = graph.model_copy(
            update={
                "warnings": list(
                    dict.fromkeys([*graph.warnings, *planning_warnings])
                )
            }
        )
        return updated, {
            "node": "planning",
            "status": "unavailable",
            "reason": "no_plannable_candidates",
            "city": resolved_city,
            "duration_days": duration_days,
            "candidate_count": len(candidates),
        }
    if planner is not None:
        try:
            proposed = planner.plan_itinerary(
                question=message,
                city=resolved_city,
                duration_days=duration_days,
                duration_nights=duration_nights,
                candidates=planner_candidates,
                weather=[item.model_dump(mode="json") for item in weather_forecast],
                latitude=latitude,
                longitude=longitude,
                personalization_context=personalization_context,
            )
            _validate_plan(
                proposed,
                candidates=candidates,
                city=resolved_city,
                duration_days=duration_days,
                duration_nights=duration_nights,
                weather_forecast=weather_forecast,
            )
            plan = proposed
            source = "gemini_structured"
        except Exception as exc:
            planning_warnings.append(
                f"planning_fallback:{exc.__class__.__name__}"
            )
            logger.warning(
                "Planning agent fallback city={} error_type={} reason={}",
                resolved_city,
                exc.__class__.__name__,
                str(exc),
            )

    if plan is None:
        try:
            plan = _fallback_plan(
                candidates,
                duration_days=duration_days,
                duration_nights=duration_nights,
                weather=_planning_weather(weather_forecast),
                latitude=latitude,
                longitude=longitude,
            )
            _validate_plan(
                plan,
                candidates=candidates,
                city=resolved_city,
                duration_days=duration_days,
                duration_nights=duration_nights,
                weather_forecast=weather_forecast,
                require_full_coverage=False,
            )
        except Exception as exc:
            planning_warnings.append(
                f"planning_fallback_unavailable:{exc.__class__.__name__}"
            )
            logger.warning(
                "Planning fallback unavailable city={} error_type={} reason={}",
                resolved_city,
                exc.__class__.__name__,
                str(exc),
            )
            updated = graph.model_copy(
                update={
                    "warnings": list(
                        dict.fromkeys([*graph.warnings, *planning_warnings])
                    )
                }
            )
            return updated, {
                "node": "planning",
                "status": "unavailable",
                "reason": "fallback_plan_invalid",
                "city": resolved_city,
                "duration_days": duration_days,
                "candidate_count": len(candidates),
            }

    itinerary = _materialize(plan, candidates)
    warnings = [*graph.warnings, *planning_warnings]
    if not weather_forecast:
        warnings.append("weather_unavailable_for_planning")
    updated = graph.model_copy(
        update={
            "answer_type": "itinerary_planning",
            "itinerary": itinerary,
            "warnings": list(dict.fromkeys(warnings)),
            "missing_fields": [
                item for item in graph.missing_fields if item != "activity"
            ],
        }
    )
    return updated, {
        "node": "planning",
        "status": "completed",
        "planner": source,
        "city": resolved_city,
        "duration_days": duration_days,
        "duration_nights": duration_nights,
        "candidate_count": len(candidates),
        "itinerary_days": len(itinerary),
        "personalized": bool(personalization_context),
    }


def requested_itinerary_duration_days(
    message: str,
    query_plan: dict[str, Any] | None = None,
) -> int:
    query_plan = query_plan or {}
    planned = query_plan.get("duration_days")
    if isinstance(planned, int) and planned > 0:
        return min(planned, 30)
    match = re.search(r"\b(\d{1,2})\s*ngay\b", normalize_text(message))
    return min(int(match.group(1)), 30) if match else 1


def _duration_nights(message: str, duration_days: int) -> int:
    match = re.search(r"\b(\d{1,2})\s*dem\b", normalize_text(message))
    if match:
        return min(int(match.group(1)), max(duration_days, 1))
    return max(0, duration_days - 1)


def _single_candidate_city(candidates: list[dict[str, Any]]) -> str | None:
    cities = list(dict.fromkeys(str(item.get("city")) for item in candidates if item.get("city")))
    return cities[0] if len(cities) == 1 else None


def _city_candidates(candidates: list[dict[str, Any]], city: str) -> list[dict[str, Any]]:
    expected = normalize_text(city)
    return [
        item
        for item in candidates
        if item.get("place_id") and normalize_text(str(item.get("city") or "")) == expected
    ]


def _validate_plan(
    plan: ItineraryPlan,
    *,
    candidates: list[dict[str, Any]],
    city: str,
    duration_days: int,
    duration_nights: int,
    weather_forecast: list[WeatherAssessment],
    require_full_coverage: bool = True,
) -> None:
    if [day.day for day in plan.days] != list(range(1, duration_days + 1)):
        raise ValueError("itinerary_day_count_or_order_invalid")
    allowed = {str(item["place_id"]): item for item in candidates}
    expected_city = normalize_text(city)
    roles: list[ItineraryRole] = []
    for day in plan.days:
        day_weather = (
            weather_forecast[day.day - 1]
            if day.day <= len(weather_forecast)
            else None
        )
        previous_end = -1
        day_roles: list[ItineraryRole] = []
        for stop in day.stops:
            candidate = allowed.get(stop.place_id)
            if candidate is None:
                raise ValueError("itinerary_contains_ungrounded_place")
            if normalize_text(str(candidate.get("city") or "")) != expected_city:
                raise ValueError("itinerary_city_scope_violation")
            start = _minutes(stop.start_time)
            end = _minutes(stop.end_time)
            if start >= end or start < previous_end:
                raise ValueError("itinerary_time_overlap")
            previous_end = end
            _validate_role(stop.role, str(candidate.get("entity_type") or ""))
            _validate_opening_window(stop, candidate)
            if (
                day_weather is not None
                and day_weather.suitability == "unsuitable"
                and stop.role == ItineraryRole.ACTIVITY
                and _is_explicitly_outdoor(candidate)
            ):
                raise ValueError("outdoor_activity_in_unsuitable_weather")
            roles.append(stop.role)
            day_roles.append(stop.role)
        if require_full_coverage and ItineraryRole.MEAL not in day_roles:
            raise ValueError("itinerary_daily_meal_missing")
    if (
        require_full_coverage
        and any(
            item.get("entity_type") in {"attraction", "nightlife"}
            for item in candidates
        )
        and ItineraryRole.ACTIVITY not in roles
    ):
        raise ValueError("itinerary_activity_missing")
    if require_full_coverage and ItineraryRole.CAFE_BREAK not in roles and ItineraryRole.REST not in roles:
        raise ValueError("itinerary_rest_missing")
    if (
        require_full_coverage
        and duration_nights > 0
        and any(item.get("entity_type") == "hotel" for item in candidates)
        and ItineraryRole.CHECK_IN not in roles
    ):
        raise ValueError("itinerary_accommodation_missing")


def _validate_role(role: ItineraryRole, entity_type: str) -> None:
    allowed = {
        ItineraryRole.ACTIVITY: {"attraction", "nightlife"},
        ItineraryRole.MEAL: {"restaurant"},
        ItineraryRole.CAFE_BREAK: {"cafe"},
        ItineraryRole.CHECK_IN: {"hotel"},
        ItineraryRole.REST: {"hotel", "cafe"},
        ItineraryRole.CHECK_OUT: {"hotel"},
    }
    if entity_type not in allowed[role]:
        raise ValueError(f"invalid_role_entity_type:{role}:{entity_type}")


def _validate_opening_window(stop: PlannedStop, candidate: dict[str, Any]) -> None:
    attributes = candidate.get("attributes") or {}
    opens = attributes.get("opening_hours_open")
    closes = attributes.get("opening_hours_close")
    if not isinstance(opens, str) or not isinstance(closes, str):
        return
    if not re.fullmatch(r"\d{1,2}:\d{2}", opens) or not re.fullmatch(r"\d{1,2}:\d{2}", closes):
        return
    opening = _minutes(opens)
    closing = 24 * 60 if closes in {"00:00", "24:00"} else _minutes(closes)
    if _minutes(stop.start_time) < opening or _minutes(stop.end_time) > closing:
        raise ValueError("itinerary_outside_opening_hours")


def _is_explicitly_outdoor(candidate: dict[str, Any]) -> bool:
    attributes = candidate.get("attributes") or {}
    indoor = attributes.get("is_indoor", attributes.get("indoor"))
    return indoor is False


def _weather_safe_candidates(
    candidates: list[dict[str, Any]],
    weather_forecast: list[WeatherAssessment],
) -> list[dict[str, Any]]:
    if not weather_forecast or any(
        item.suitability != "unsuitable" for item in weather_forecast
    ):
        return candidates
    return [
        item
        for item in candidates
        if item.get("entity_type") not in {"attraction", "nightlife"}
        or not _is_explicitly_outdoor(item)
    ]


def _fallback_plan(
    candidates: list[dict[str, Any]],
    *,
    duration_days: int,
    duration_nights: int,
    weather: WeatherAssessment | None,
    latitude: float | None,
    longitude: float | None,
) -> ItineraryPlan:
    pools = {
        entity_type: _rank_candidates(
            [
                item
                for item in candidates
                if item.get("entity_type") == entity_type
                and not (
                    entity_type == "attraction"
                    and weather is not None
                    and weather.suitability == "unsuitable"
                    and _is_explicitly_outdoor(item)
                )
            ],
            weather=weather,
            latitude=latitude,
            longitude=longitude,
        )
        for entity_type in ("attraction", "restaurant", "cafe", "hotel", "nightlife")
    }
    used: set[str] = set()

    def take(entity_type: str, *, reusable: bool = False) -> dict[str, Any] | None:
        for item in pools[entity_type]:
            place_id = str(item["place_id"])
            if reusable or place_id not in used:
                if not reusable:
                    used.add(place_id)
                return item
        return pools[entity_type][0] if reusable and pools[entity_type] else None

    hotel = take("hotel", reusable=True) if duration_nights else None
    days: list[PlannedDay] = []
    for day_number in range(1, duration_days + 1):
        specs: list[tuple[str, str, ItineraryRole, dict[str, Any] | None]] = []
        if day_number > 1 and hotel is not None:
            specs.append(("08:00", "08:30", ItineraryRole.CHECK_OUT, hotel))
        specs.extend(
            [
                ("09:00", "11:00", ItineraryRole.ACTIVITY, take("attraction")),
                ("11:30", "13:00", ItineraryRole.MEAL, take("restaurant")),
                ("14:00", "16:00", ItineraryRole.ACTIVITY, take("attraction")),
                ("16:30", "17:30", ItineraryRole.CAFE_BREAK, take("cafe")),
                ("18:00", "19:30", ItineraryRole.MEAL, take("restaurant")),
            ]
        )
        if day_number <= duration_nights and hotel is not None:
            specs.append(("20:00", "20:30", ItineraryRole.CHECK_IN, hotel))
        stops = [
            PlannedStop(
                start_time=start,
                end_time=end,
                place_id=str(item["place_id"]),
                role=role,
                rationale=_fallback_rationale(role, weather),
            )
            for start, end, role, item in specs
            if item is not None
        ]
        if not stops:
            for entity_type, role, start, end in (
                ("attraction", ItineraryRole.ACTIVITY, "09:00", "11:00"),
                ("nightlife", ItineraryRole.ACTIVITY, "18:00", "20:00"),
                ("restaurant", ItineraryRole.MEAL, "12:00", "13:00"),
                ("cafe", ItineraryRole.CAFE_BREAK, "16:30", "17:30"),
                ("hotel", ItineraryRole.REST, "14:00", "15:00"),
            ):
                item = take(entity_type, reusable=True)
                if item is None:
                    continue
                if (
                    role == ItineraryRole.ACTIVITY
                    and weather is not None
                    and weather.suitability == "unsuitable"
                    and _is_explicitly_outdoor(item)
                ):
                    continue
                stops.append(
                    PlannedStop(
                        start_time=start,
                        end_time=end,
                        place_id=str(item["place_id"]),
                        role=role,
                        rationale=_fallback_rationale(role, weather),
                    )
                )
                break
        days.append(PlannedDay(day=day_number, stops=stops))
    return ItineraryPlan(days=days, summary="Lịch trình cân bằng từ các địa điểm đã xác minh.")


def _has_fallback_candidates(
    candidates: list[dict[str, Any]],
    *,
    duration_nights: int,
) -> bool:
    return any(
        item.get("entity_type") in PLANNABLE_ENTITY_TYPES
        and (
            item.get("entity_type") != "hotel"
            or duration_nights > 0
        )
        for item in candidates
    )


def _planning_weather(
    forecast: list[WeatherAssessment],
) -> WeatherAssessment | None:
    return next(
        (item for item in forecast if item.suitability == "unsuitable"),
        forecast[0] if forecast else None,
    )


def _rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    weather: WeatherAssessment | None,
    latitude: float | None,
    longitude: float | None,
) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[float, float, str]:
        outdoor_penalty = 1.0 if (
            weather is not None
            and weather.suitability == "unsuitable"
            and _is_explicitly_outdoor(item)
        ) else 0.0
        distance = _distance_from_origin(item, latitude, longitude)
        score = float(item.get("score") or 0)
        return outdoor_penalty, distance - score, str(item.get("place_id"))

    return sorted(candidates, key=key)


def _distance_from_origin(
    candidate: dict[str, Any],
    latitude: float | None,
    longitude: float | None,
) -> float:
    if latitude is None or longitude is None:
        return float(candidate.get("distance_km") or 9999)
    attributes = candidate.get("attributes") or {}
    target_lat = attributes.get("lat", attributes.get("latitude"))
    target_lon = attributes.get("lng", attributes.get("longitude"))
    if not isinstance(target_lat, (int, float)) or not isinstance(target_lon, (int, float)):
        return 9999
    radius_km = 6371.0088
    lat1, lat2 = math.radians(latitude), math.radians(float(target_lat))
    delta_lat = lat2 - lat1
    delta_lon = math.radians(float(target_lon) - longitude)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _fallback_rationale(role: ItineraryRole, weather: WeatherAssessment | None) -> str:
    weather_note = " Có cân nhắc điều kiện thời tiết hiện tại." if weather else ""
    labels = {
        ItineraryRole.ACTIVITY: "Điểm vui chơi/tham quan được xác minh trong Knowledge Base.",
        ItineraryRole.MEAL: "Khung giờ ăn uống được bố trí giữa các hoạt động.",
        ItineraryRole.CAFE_BREAK: "Khoảng nghỉ nhẹ để lịch trình không quá dày.",
        ItineraryRole.CHECK_IN: "Khung giờ nhận phòng và nghỉ ngơi.",
        ItineraryRole.REST: "Khoảng nghỉ trong lịch trình.",
        ItineraryRole.CHECK_OUT: "Khung giờ trả phòng trước hoạt động trong ngày.",
    }
    return labels[role] + weather_note


def _materialize(plan: ItineraryPlan, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item["place_id"]): item for item in candidates}
    return [
        {
            "day": day.day,
            "slots": [
                {
                    "order": index,
                    "start_time": stop.start_time,
                    "end_time": stop.end_time,
                    "place_id": stop.place_id,
                    "name": by_id[stop.place_id].get("name"),
                    "city": by_id[stop.place_id].get("city"),
                    "entity_type": by_id[stop.place_id].get("entity_type"),
                    "role": stop.role.value,
                    "rationale": stop.rationale,
                }
                for index, stop in enumerate(day.stops, start=1)
            ],
        }
        for day in plan.days
    ]


def _minutes(value: str) -> int:
    hours, minutes = value.split(":", 1)
    return int(hours) * 60 + int(minutes)


__all__ = [
    "ItineraryPlan",
    "ItineraryRole",
    "PlannedDay",
    "PlannedStop",
    "SupportsItineraryPlanning",
    "is_itinerary_request",
    "planning_agent_node",
    "requested_itinerary_duration_days",
]
