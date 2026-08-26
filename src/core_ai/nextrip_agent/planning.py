from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol
from zoneinfo import ZoneInfo

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


class PreferredPeriod(StrEnum):
    MORNING = "morning"
    LUNCH = "lunch"
    AFTERNOON = "afternoon"
    DINNER = "dinner"
    EVENING = "evening"
    FLEXIBLE = "flexible"


class SemanticRole(StrEnum):
    ACTIVITY = "activity"
    MEAL = "meal"
    CAFE_BREAK = "cafe_break"
    REST = "rest"


class PlannedStop(BaseModel):
    start_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    place_id: str = Field(min_length=1)
    role: ItineraryRole
    rationale: str = Field(min_length=1, max_length=300)
    transport_to_next: dict[str, Any] | None = None


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


class SemanticAssignment(BaseModel):
    """A grounded choice from Gemini; Python owns the actual clock schedule."""

    place_id: str = Field(min_length=1)
    role: SemanticRole
    preferred_period: PreferredPeriod = PreferredPeriod.FLEXIBLE
    rationale: str = Field(min_length=1, max_length=300)


class SemanticDay(BaseModel):
    day: int = Field(ge=1, le=30)
    assignments: list[SemanticAssignment] = Field(min_length=1, max_length=8)


class SemanticStay(BaseModel):
    hotel_id: str = Field(min_length=1)


class SemanticItineraryPlan(BaseModel):
    days: list[SemanticDay] = Field(min_length=1, max_length=30)
    stay: SemanticStay | None = None
    summary: str = Field(default="", max_length=500)


class SemanticAssignmentDraft(BaseModel):
    """Gemini-facing semantic schema without unsupported JSON Schema bounds."""

    place_id: str
    role: SemanticRole
    preferred_period: PreferredPeriod = PreferredPeriod.FLEXIBLE
    rationale: str


class SemanticDayDraft(BaseModel):
    day: int
    assignments: list[SemanticAssignmentDraft]


class SemanticStayDraft(BaseModel):
    hotel_id: str


class SemanticItineraryPlanDraft(BaseModel):
    days: list[SemanticDayDraft]
    stay: SemanticStayDraft | None = None
    summary: str = ""


class PlanningPolicy(BaseModel):
    """Configurable scheduling policy; it is not an itinerary template."""

    model_config = {"frozen": True}

    day_start_minutes: int = Field(default=8 * 60, ge=0, lt=24 * 60)
    day_end_minutes: int = Field(default=21 * 60 + 30, gt=0, le=24 * 60)
    morning_end_minutes: int = Field(default=11 * 60 + 30, gt=0, le=24 * 60)
    lunch_start_minutes: int = Field(default=11 * 60 + 15, ge=0, lt=24 * 60)
    lunch_end_minutes: int = Field(default=14 * 60, gt=0, le=24 * 60)
    afternoon_start_minutes: int = Field(default=13 * 60 + 30, ge=0, lt=24 * 60)
    afternoon_end_minutes: int = Field(default=18 * 60, gt=0, le=24 * 60)
    dinner_start_minutes: int = Field(default=17 * 60 + 30, ge=0, lt=24 * 60)
    dinner_end_minutes: int = Field(default=20 * 60 + 30, gt=0, le=24 * 60)
    evening_start_minutes: int = Field(default=18 * 60 + 30, ge=0, lt=24 * 60)
    transition_buffer_minutes: int = Field(default=15, ge=0, le=120)
    hotel_transition_minutes: int = Field(default=30, ge=10, le=180)
    activity_duration_minutes: int = Field(default=90, ge=30, le=360)
    meal_duration_minutes: int = Field(default=75, ge=30, le=240)
    cafe_duration_minutes: int = Field(default=60, ge=20, le=180)
    rest_duration_minutes: int = Field(default=60, ge=20, le=240)


DEFAULT_PLANNING_POLICY = PlanningPolicy()
_VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


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


class SupportsSemanticItineraryPlanning(Protocol):
    def plan_semantic_itinerary(
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
    ) -> SemanticItineraryPlan: ...


class SupportsPlanningRoutes(Protocol):
    def recommend_transport(
        self,
        *,
        origin_id: str,
        destination_id: str,
        departure_time: datetime,
    ) -> dict[str, Any]: ...


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
    policy: PlanningPolicy = DEFAULT_PLANNING_POLICY,
    route_provider: SupportsPlanningRoutes | None = None,
    travel_date: date | None = None,
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
    # Keep explicit exclusions out of the candidate pool before ranking.  A
    # nightlife result should never re-enter an itinerary that explicitly
    # rejects bars or clubs merely because it scored highly in retrieval.
    normalized_message = normalize_text(message)
    if any(term in normalized_message for term in ("khong bar", "khong club", "khong muon di bar", "khong muon di club")):
        candidates = [
            item for item in candidates if item.get("entity_type") != "nightlife"
        ]
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
            semantic_method = getattr(planner, "plan_semantic_itinerary", None)
            planning_arguments = {
                "question": message,
                "city": resolved_city,
                "duration_days": duration_days,
                "duration_nights": duration_nights,
                "candidates": planner_candidates,
                "weather": [
                    item.model_dump(mode="json") for item in weather_forecast
                ],
                "latitude": latitude,
                "longitude": longitude,
                "personalization_context": personalization_context,
            }
            if callable(semantic_method):
                semantic_plan = semantic_method(**planning_arguments)
                proposed = _schedule_semantic_plan(
                    semantic_plan,
                    candidates=planner_candidates,
                    duration_days=duration_days,
                    duration_nights=duration_nights,
                    policy=policy,
                    route_provider=route_provider,
                    travel_date=travel_date,
                )
                source = "gemini_hybrid"
            else:
                proposed = planner.plan_itinerary(**planning_arguments)
                source = "gemini_structured"
            _validate_plan(
                proposed,
                candidates=candidates,
                city=resolved_city,
                duration_days=duration_days,
                duration_nights=duration_nights,
                weather_forecast=weather_forecast,
                policy=policy,
            )
            plan = proposed
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
                message=message,
                policy=policy,
                route_provider=route_provider,
                travel_date=travel_date,
            )
            _validate_plan(
                plan,
                candidates=candidates,
                city=resolved_city,
                duration_days=duration_days,
                duration_nights=duration_nights,
                weather_forecast=weather_forecast,
                require_full_coverage=False,
                policy=policy,
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
    policy: PlanningPolicy = DEFAULT_PLANNING_POLICY,
) -> None:
    if [day.day for day in plan.days] != list(range(1, duration_days + 1)):
        raise ValueError("itinerary_day_count_or_order_invalid")
    allowed = {str(item["place_id"]): item for item in candidates}
    expected_city = normalize_text(city)
    roles: list[ItineraryRole] = []
    role_locations: list[tuple[int, PlannedStop]] = []
    activity_candidate_count = sum(
        item.get("entity_type") in {"attraction", "nightlife"}
        for item in candidates
    )
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
            if (
                start >= end
                or start < previous_end
                or start < policy.day_start_minutes
                or end > policy.day_end_minutes
            ):
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
            role_locations.append((day.day, stop))
        if require_full_coverage and ItineraryRole.MEAL not in day_roles:
            raise ValueError("itinerary_daily_meal_missing")
        if (
            require_full_coverage
            and activity_candidate_count >= duration_days
            and ItineraryRole.ACTIVITY not in day_roles
        ):
            raise ValueError("itinerary_daily_activity_missing")
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
        and any(
            item.get("entity_type") == "hotel"
            and _hotel_candidate_is_selectable(item)
            for item in candidates
        )
        and ItineraryRole.CHECK_IN not in roles
    ):
        raise ValueError("itinerary_accommodation_missing")
    _validate_hotel_lifecycle(
        role_locations,
        candidates=candidates,
        duration_days=duration_days,
        duration_nights=duration_nights,
        require_full_coverage=require_full_coverage,
    )


def _validate_hotel_lifecycle(
    role_locations: list[tuple[int, PlannedStop]],
    *,
    candidates: list[dict[str, Any]],
    duration_days: int,
    duration_nights: int,
    require_full_coverage: bool,
) -> None:
    if duration_nights <= 0 or not any(
        item.get("entity_type") == "hotel"
        and _hotel_candidate_is_selectable(item)
        for item in candidates
    ):
        return
    check_ins = [
        (day, stop)
        for day, stop in role_locations
        if stop.role == ItineraryRole.CHECK_IN
    ]
    check_outs = [
        (day, stop)
        for day, stop in role_locations
        if stop.role == ItineraryRole.CHECK_OUT
    ]
    if len(check_ins) != 1:
        raise ValueError("itinerary_hotel_check_in_count_invalid")
    if duration_days > 1 and len(check_outs) != 1:
        raise ValueError("itinerary_hotel_check_out_count_invalid")
    if check_ins[0][0] != 1:
        raise ValueError("itinerary_hotel_check_in_day_invalid")
    if duration_days > 1:
        if check_outs[0][0] != duration_days:
            raise ValueError("itinerary_hotel_check_out_day_invalid")
        if check_ins[0][1].place_id != check_outs[0][1].place_id:
            raise ValueError("itinerary_hotel_stay_identity_mismatch")
    elif require_full_coverage and check_outs:
        raise ValueError("itinerary_same_day_hotel_checkout_invalid")


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
    message: str = "",
    policy: PlanningPolicy = DEFAULT_PLANNING_POLICY,
    route_provider: SupportsPlanningRoutes | None = None,
    travel_date: date | None = None,
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
            message=message,
        )
        for entity_type in ("attraction", "restaurant", "cafe", "hotel", "nightlife")
    }
    activity_pool = [*pools["attraction"], *pools["nightlife"]]
    indexes = {entity_type: 0 for entity_type in pools}
    activity_index = 0

    def take(entity_type: str, *, required: bool = False) -> dict[str, Any] | None:
        values = pools[entity_type]
        index = indexes[entity_type]
        if index < len(values):
            indexes[entity_type] += 1
            return values[index]
        return values[0] if required and values else None

    def take_activity(*, required: bool = False) -> dict[str, Any] | None:
        nonlocal activity_index
        if activity_index < len(activity_pool):
            value = activity_pool[activity_index]
            activity_index += 1
            return value
        return activity_pool[0] if required and activity_pool else None

    selectable_hotels = [
        item for item in pools["hotel"] if _hotel_candidate_is_selectable(item)
    ]
    hotel = selectable_hotels[0] if duration_nights and selectable_hotels else None
    semantic_days: list[SemanticDay] = []
    for day_number in range(1, duration_days + 1):
        assignments: list[SemanticAssignment] = []

        def add(
            item: dict[str, Any] | None,
            role: ItineraryRole,
            period: PreferredPeriod,
        ) -> None:
            if item is not None:
                assignments.append(
                    SemanticAssignment(
                        place_id=str(item["place_id"]),
                        role=SemanticRole(role.value),
                        preferred_period=period,
                        rationale=_fallback_rationale(role, weather),
                    )
                )

        add(take_activity(required=True), ItineraryRole.ACTIVITY, PreferredPeriod.MORNING)
        add(take("restaurant", required=True), ItineraryRole.MEAL, PreferredPeriod.LUNCH)
        # Preserve enough primary activities for later days before adding an
        # optional second stop to the current day.
        remaining_days = duration_days - day_number
        remaining_unique_activities = len(activity_pool) - activity_index
        if remaining_unique_activities > remaining_days:
            add(take_activity(), ItineraryRole.ACTIVITY, PreferredPeriod.AFTERNOON)
        add(take("cafe", required=True), ItineraryRole.CAFE_BREAK, PreferredPeriod.AFTERNOON)
        if len(pools["restaurant"]) - indexes["restaurant"] > remaining_days:
            add(take("restaurant"), ItineraryRole.MEAL, PreferredPeriod.DINNER)
        if not assignments and hotel is not None:
            add(hotel, ItineraryRole.REST, PreferredPeriod.AFTERNOON)
        if assignments:
            semantic_days.append(SemanticDay(day=day_number, assignments=assignments))

    semantic = SemanticItineraryPlan(
        days=semantic_days,
        stay=SemanticStay(hotel_id=str(hotel["place_id"])) if hotel is not None else None,
        summary="Lịch trình cân bằng từ các địa điểm đã xác minh.",
    )
    return _schedule_semantic_plan(
        semantic,
        candidates=candidates,
        duration_days=duration_days,
        duration_nights=duration_nights,
        policy=policy,
        allow_repeated_places=True,
        route_provider=route_provider,
        travel_date=travel_date,
    )


def _schedule_semantic_plan(
    semantic: SemanticItineraryPlan,
    *,
    candidates: list[dict[str, Any]],
    duration_days: int,
    duration_nights: int,
    policy: PlanningPolicy,
    allow_repeated_places: bool = False,
    route_provider: SupportsPlanningRoutes | None = None,
    travel_date: date | None = None,
) -> ItineraryPlan:
    if [day.day for day in semantic.days] != list(range(1, duration_days + 1)):
        raise ValueError("semantic_itinerary_day_count_or_order_invalid")
    by_id = {str(item["place_id"]): item for item in candidates}
    stay_hotel = _resolve_stay_hotel(
        semantic,
        candidates=candidates,
        duration_nights=duration_nights,
    )
    used: set[str] = set()
    planned_days: list[PlannedDay] = []
    for semantic_day in semantic.days:
        stops: list[PlannedStop] = []
        cursor = policy.day_start_minutes
        local_date = (
            travel_date + timedelta(days=semantic_day.day - 1)
            if travel_date is not None
            else None
        )
        if stay_hotel is not None and duration_days > 1 and semantic_day.day == duration_days:
            checkout_end = cursor + policy.hotel_transition_minutes
            stops.append(
                PlannedStop(
                    start_time=_clock(cursor),
                    end_time=_clock(checkout_end),
                    place_id=str(stay_hotel["place_id"]),
                    role=ItineraryRole.CHECK_OUT,
                    rationale="Trả phòng vào ngày cuối của kỳ lưu trú.",
                )
            )
            cursor = checkout_end

        for assignment in semantic_day.assignments:
            role = ItineraryRole(assignment.role.value)
            candidate = by_id.get(assignment.place_id)
            if candidate is None:
                raise ValueError("semantic_itinerary_contains_ungrounded_place")
            _validate_role(role, str(candidate.get("entity_type") or ""))
            if not allow_repeated_places and assignment.place_id in used:
                raise ValueError("semantic_itinerary_repeats_place")
            used.add(assignment.place_id)
            period_start, period_end = _period_window(assignment.preferred_period, policy)
            opening, closing = _opening_window(candidate, policy)
            duration = _visit_duration_minutes(candidate, role, policy)
            cursor = _ready_after_previous_stop(
                stops,
                destination_id=assignment.place_id,
                local_date=local_date,
                route_provider=route_provider,
                policy=policy,
            )
            start = max(cursor, period_start, opening)
            latest_end = min(policy.day_end_minutes, closing)
            # The preferred period is soft, but lunch/dinner remain bounded so
            # a meal cannot silently drift into a different part of the day.
            if assignment.preferred_period in {
                PreferredPeriod.LUNCH,
                PreferredPeriod.DINNER,
            }:
                latest_end = min(latest_end, period_end)
            end = start + duration
            if end > latest_end:
                raise ValueError(
                    f"semantic_assignment_not_schedulable:{assignment.place_id}"
                )
            stops.append(
                PlannedStop(
                    start_time=_clock(start),
                    end_time=_clock(end),
                    place_id=assignment.place_id,
                    role=role,
                    rationale=assignment.rationale,
                )
            )
            cursor = end

        if stay_hotel is not None and semantic_day.day == 1:
            opening, closing = _opening_window(stay_hotel, policy)
            cursor = _ready_after_previous_stop(
                stops,
                destination_id=str(stay_hotel["place_id"]),
                local_date=local_date,
                route_provider=route_provider,
                policy=policy,
            )
            checkin_start = max(cursor, opening)
            checkin_end = checkin_start + policy.hotel_transition_minutes
            if checkin_end > min(closing, policy.day_end_minutes):
                raise ValueError("hotel_check_in_not_schedulable")
            stops.append(
                PlannedStop(
                    start_time=_clock(checkin_start),
                    end_time=_clock(checkin_end),
                    place_id=str(stay_hotel["place_id"]),
                    role=ItineraryRole.CHECK_IN,
                    rationale="Nhận phòng một lần cho toàn bộ kỳ lưu trú.",
                )
            )
        if not stops:
            raise ValueError("semantic_itinerary_day_empty")
        planned_days.append(PlannedDay(day=semantic_day.day, stops=stops))
    return ItineraryPlan(days=planned_days, summary=semantic.summary)


def _ready_after_previous_stop(
    stops: list[PlannedStop],
    *,
    destination_id: str,
    local_date: date | None,
    route_provider: SupportsPlanningRoutes | None,
    policy: PlanningPolicy,
) -> int:
    if not stops:
        return policy.day_start_minutes
    origin = stops[-1]
    origin_end = _minutes(origin.end_time)
    fallback = origin_end + policy.transition_buffer_minutes
    if (
        route_provider is None
        or local_date is None
        or origin.place_id == destination_id
    ):
        return fallback
    departure = datetime.combine(
        local_date,
        datetime.strptime(origin.end_time, "%H:%M").time(),
        tzinfo=_VIETNAM_TIMEZONE,
    )
    try:
        payload = route_provider.recommend_transport(
            origin_id=origin.place_id,
            destination_id=destination_id,
            departure_time=departure,
        )
        summary = _planning_transport_summary(
            payload,
            origin_id=origin.place_id,
            destination_id=destination_id,
            departure=departure,
        )
        duration_seconds = summary.get("duration_seconds")
        if not isinstance(duration_seconds, int) or duration_seconds <= 0:
            return fallback
        origin.transport_to_next = summary
        travel_minutes = math.ceil(duration_seconds / 60)
        return origin_end + travel_minutes + policy.transition_buffer_minutes
    except Exception as exc:
        logger.warning(
            "Planning route degraded origin={} destination={} error_type={}",
            origin.place_id,
            destination_id,
            exc.__class__.__name__,
        )
        return fallback


def _planning_transport_summary(
    payload: dict[str, Any],
    *,
    origin_id: str,
    destination_id: str,
    departure: datetime,
) -> dict[str, Any]:
    options = payload.get("options")
    normalized_options = options if isinstance(options, list) else []
    selected = next(
        (
            option
            for option in normalized_options
            if isinstance(option, dict) and option.get("recommended") is True
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise ValueError("planning route has no recommended option")
    route_response = selected.get("route")
    route = (
        route_response.get("route")
        if isinstance(route_response, dict)
        and isinstance(route_response.get("route"), dict)
        else {}
    )
    duration_seconds = selected.get("duration_seconds")
    if not isinstance(duration_seconds, int) or duration_seconds <= 0:
        raise ValueError("planning route has no positive duration")
    return {
        "status": str(payload.get("status") or "unavailable"),
        "origin_place_id": origin_id,
        "destination_place_id": destination_id,
        "departure_time": departure.isoformat(),
        "recommended_mode": payload.get("recommended_mode"),
        "distance_meters": selected.get("distance_meters"),
        "duration_seconds": duration_seconds,
        "provider": route.get("provider"),
        "traffic_basis": route.get("traffic_basis"),
        "traffic_aware": route.get("traffic_aware"),
        "degraded": bool(payload.get("degraded", False)),
        "partial": bool(payload.get("partial", False)),
        "selection_reason": payload.get("selection_reason"),
        "alternatives": normalized_options,
        "error_code": None,
    }


def _resolve_stay_hotel(
    semantic: SemanticItineraryPlan,
    *,
    candidates: list[dict[str, Any]],
    duration_nights: int,
) -> dict[str, Any] | None:
    if duration_nights <= 0:
        return None
    hotels = [item for item in candidates if item.get("entity_type") == "hotel"]
    if not hotels:
        return None
    if semantic.stay is None:
        return next((item for item in hotels if _hotel_candidate_is_selectable(item)), None)
    selected = next(
        (item for item in hotels if item.get("place_id") == semantic.stay.hotel_id),
        None,
    )
    if selected is None:
        raise ValueError("semantic_stay_contains_ungrounded_hotel")
    if not _hotel_candidate_is_selectable(selected):
        raise ValueError("semantic_stay_hotel_unavailable")
    return selected


def _hotel_candidate_is_selectable(candidate: dict[str, Any]) -> bool:
    availability = (candidate.get("attributes") or {}).get("hotel_availability")
    if not isinstance(availability, dict):
        return True
    windows = availability.get("windows")
    # Explicit operational evidence takes precedence over semantic ranking.
    # Missing evidence remains eligible so the planner can still degrade
    # gracefully when Current Data is temporarily unavailable.
    if isinstance(windows, list) and windows:
        return isinstance(availability.get("selected_window_index"), int)
    return True


def _period_window(
    period: PreferredPeriod,
    policy: PlanningPolicy,
) -> tuple[int, int]:
    windows = {
        PreferredPeriod.MORNING: (
            policy.day_start_minutes,
            policy.morning_end_minutes,
        ),
        PreferredPeriod.LUNCH: (
            policy.lunch_start_minutes,
            policy.lunch_end_minutes,
        ),
        PreferredPeriod.AFTERNOON: (
            policy.afternoon_start_minutes,
            policy.afternoon_end_minutes,
        ),
        PreferredPeriod.DINNER: (
            policy.dinner_start_minutes,
            policy.dinner_end_minutes,
        ),
        PreferredPeriod.EVENING: (
            policy.evening_start_minutes,
            policy.day_end_minutes,
        ),
        PreferredPeriod.FLEXIBLE: (policy.day_start_minutes, policy.day_end_minutes),
    }
    return windows[period]


def _opening_window(
    candidate: dict[str, Any],
    policy: PlanningPolicy,
) -> tuple[int, int]:
    attributes = dict(candidate.get("attributes") or {})
    current = attributes.get("current")
    if isinstance(current, dict):
        attributes = {**attributes, **current}
    opens = attributes.get("opening_hours_open")
    closes = attributes.get("opening_hours_close")
    if not isinstance(opens, str) or not re.fullmatch(r"\d{1,2}:\d{2}", opens):
        opening = policy.day_start_minutes
    else:
        opening = _minutes(opens)
    if not isinstance(closes, str) or not re.fullmatch(r"\d{1,2}:\d{2}", closes):
        closing = policy.day_end_minutes
    else:
        closing = 24 * 60 if closes in {"00:00", "24:00"} else _minutes(closes)
        if closing <= opening:
            closing = policy.day_end_minutes
    return opening, closing


def _visit_duration_minutes(
    candidate: dict[str, Any],
    role: ItineraryRole,
    policy: PlanningPolicy,
) -> int:
    attributes = candidate.get("attributes") or {}
    explicit = attributes.get("duration_recommendation")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return min(explicit, 6 * 60)
    if isinstance(explicit, str):
        numbers = [int(value) for value in re.findall(r"\d+", explicit)]
        if numbers:
            value = sum(numbers[:2]) // min(len(numbers), 2)
            if "giờ" in explicit.casefold() or "hour" in explicit.casefold():
                value *= 60
            return max(20, min(value, 6 * 60))
    defaults = {
        ItineraryRole.ACTIVITY: policy.activity_duration_minutes,
        ItineraryRole.MEAL: policy.meal_duration_minutes,
        ItineraryRole.CAFE_BREAK: policy.cafe_duration_minutes,
        ItineraryRole.REST: policy.rest_duration_minutes,
    }
    return defaults[role]


def _clock(total_minutes: int) -> str:
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


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
    message: str = "",
) -> list[dict[str, Any]]:
    normalized_message = normalize_text(message)
    compact_pace = any(
        term in normalized_message
        for term in ("nguoi lon tuoi", "nguoi cao tuoi", "nhe nhang", "khong qua nhieu")
    )
    budget_focus = any(term in normalized_message for term in ("tiet kiem", "re", "ngan sach", "budget"))
    rain_focus = any(term in normalized_message for term in ("mua", "mua phun", "thoi tiet xau", "rain"))
    beach_focus = any(term in normalized_message for term in ("bien", "bai tam", "beach", "ven bien"))
    culture_focus = any(term in normalized_message for term in ("van hoa", "lich su", "bao tang", "di tich", "chua", "thap"))
    food_focus = any(term in normalized_message for term in ("am thuc", "an uong", "mon ngon", "quan an"))
    family_focus = any(term in normalized_message for term in ("tre nho", "gia dinh"))
    excluded_nightlife = any(term in normalized_message for term in ("khong bar", "khong club", "khong muon di bar", "khong muon di club"))

    def key(item: dict[str, Any]) -> tuple[float, float, str]:
        attributes = item.get("attributes") or {}
        searchable = normalize_text(" ".join(
            str(item.get(key) or "") for key in ("name", "category", "description")
        ) + " " + " ".join(str(value) for value in attributes.values()))
        outdoor_penalty = 0.0
        if weather is not None and weather.suitability == "unsuitable" and _is_explicitly_outdoor(item):
            outdoor_penalty += 10.0
        elif rain_focus:
            if _is_explicitly_outdoor(item):
                outdoor_penalty += 2.0
            if attributes.get("is_indoor") is True or attributes.get("indoor") is True or "all weather" in searchable or "light rain" in searchable:
                outdoor_penalty -= 2.0
        preference_bonus = 0.0
        entity_type = str(item.get("entity_type") or "")
        if excluded_nightlife and entity_type == "nightlife":
            preference_bonus += 100.0
        if beach_focus and any(term in searchable for term in ("bien", "beach", "sea", "coast", "bai tam", "ven bien")):
            preference_bonus -= 4.0
        if culture_focus and any(term in searchable for term in ("van hoa", "lich su", "museum", "bao tang", "chua", "thap", "di tich")):
            preference_bonus -= 4.0
        if food_focus and entity_type == "restaurant":
            preference_bonus -= 3.0
        if family_focus and any(term in searchable for term in ("family", "gia dinh", "tre em", "children")):
            preference_bonus -= 2.0
        if compact_pace and any(term in searchable for term in ("museum", "bao tang", "park", "cafe", "spa", "hot spring", "suoi khoang")):
            preference_bonus -= 1.5
        if budget_focus:
            maximum = attributes.get("price_per_person_max")
            try:
                preference_bonus += min(float(maximum) / 1_000_000, 5.0) if maximum is not None else 0.0
            except (TypeError, ValueError):
                pass
        distance = _distance_from_origin(item, latitude, longitude)
        score = float(item.get("score") or 0)
        return outdoor_penalty + preference_bonus, distance - score, str(item.get("place_id"))

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
                    "transport_to_next": stop.transport_to_next,
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
    "DEFAULT_PLANNING_POLICY",
    "ItineraryPlan",
    "ItineraryRole",
    "PlanningPolicy",
    "PlannedDay",
    "PlannedStop",
    "PreferredPeriod",
    "SemanticAssignment",
    "SemanticDay",
    "SemanticItineraryPlan",
    "SemanticItineraryPlanDraft",
    "SemanticRole",
    "SemanticStay",
    "SupportsItineraryPlanning",
    "SupportsSemanticItineraryPlanning",
    "is_itinerary_request",
    "planning_agent_node",
    "requested_itinerary_duration_days",
]
