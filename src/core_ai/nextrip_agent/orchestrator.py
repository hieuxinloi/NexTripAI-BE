from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from enum import Enum
from time import perf_counter
from typing import Any

from src.core_ai.nextrip_agent.constants import KbVersion
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.core_ai.nextrip_agent.nodes.knowledge import SupportsKbSearch
from src.core_ai.nextrip_agent.planning import (
    SupportsItineraryPlanning,
    is_itinerary_request,
    planning_agent_node,
    requested_itinerary_duration_days,
)
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.weather import (
    WEATHER_TERMS,
    WeatherAgent,
    WeatherAssessment,
    normalize_text,
    supported_city_from_coordinates,
)
from src.infra.weather import (
    OpenMeteoWeatherClient,
    WeatherLocationRequired,
    WeatherUnavailable,
)
from src.infra.routes import (
    RouteUnavailable,
    RouteWaypoint,
    SupportsRoutes,
)


GRAPH_INTENT_TERMS = {
    "goi y",
    "dia diem",
    "di dau",
    "di choi",
    "tham quan",
    "lich trinh",
    "an gi",
    "quan an",
    "nha hang",
    "cafe",
    "khach san",
    "gan day",
}

GREETING_MESSAGES = {
    "alo",
    "chao",
    "chao ban",
    "hello",
    "hello nextrip",
    "hey",
    "hi",
    "hi nextrip",
    "xin chao",
    "xin chao nextrip",
}

GREETING_ANSWER = (
    "Xin chào! Mình là NexTripAI. Bạn muốn khám phá Quy Nhơn hay Đà Nẵng? "
    "Mình có thể gợi ý địa điểm, ăn uống, lưu trú hoặc lên lịch trình phù hợp."
)


class OrchestrationMode(str, Enum):
    CONVERSATION = "conversation"
    GRAPH_ONLY = "graph_only"
    WEATHER_ONLY = "weather_only"
    GRAPH_AND_WEATHER = "graph_and_weather"
    ITINERARY_PLANNING = "itinerary_planning"
    GRAPH_AND_ROUTE = "graph_and_route"


@dataclass(frozen=True)
class OrchestrationPlan:
    mode: OrchestrationMode
    run_graph: bool
    run_weather: bool
    run_planning: bool
    reason: str

    def trace_event(self) -> dict[str, Any]:
        return {
            "node": "orchestrator",
            "status": "planned",
            "mode": self.mode.value,
            "run_graph": self.run_graph,
            "run_weather": self.run_weather,
            "run_planning": self.run_planning,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OrchestratedResult:
    plan: OrchestrationPlan
    graph: AgentResult
    weather: WeatherAssessment | None
    weather_forecast: list[WeatherAssessment]
    weather_trace: dict[str, Any]
    planning_trace: dict[str, Any]
    route_trace: dict[str, Any]
    trace: list[dict[str, Any]]


def build_orchestration_plan(
    *,
    message: str,
    travel_date: date | None,
    include_weather: bool | None,
    entity_types: list[str] | None,
) -> OrchestrationPlan:
    if _is_greeting(message):
        return OrchestrationPlan(
            mode=OrchestrationMode.CONVERSATION,
            run_graph=False,
            run_weather=False,
            run_planning=False,
            reason="greeting",
        )
    itinerary_requested = is_itinerary_request(message)
    if itinerary_requested:
        return OrchestrationPlan(
            mode=OrchestrationMode.ITINERARY_PLANNING,
            run_graph=True,
            run_weather=include_weather is not False,
            run_planning=True,
            reason="grounded_weather_aware_itinerary",
        )
    run_weather = WeatherAgent.should_run(
        message=message,
        travel_date=travel_date,
        include_weather=include_weather,
        required_tools=[],
    )
    normalized = normalize_text(message)
    has_weather_language = any(term in normalized for term in WEATHER_TERMS)
    has_graph_language = bool(entity_types) or any(
        term in normalized for term in GRAPH_INTENT_TERMS
    )
    weather_only = run_weather and has_weather_language and not has_graph_language
    if weather_only:
        return OrchestrationPlan(
            mode=OrchestrationMode.WEATHER_ONLY,
            run_graph=False,
            run_weather=True,
            run_planning=False,
            reason="explicit_weather_query",
        )
    if run_weather:
        return OrchestrationPlan(
            mode=OrchestrationMode.GRAPH_AND_WEATHER,
            run_graph=True,
            run_weather=True,
            run_planning=False,
            reason="travel_query_with_weather_context",
        )
    return OrchestrationPlan(
        mode=OrchestrationMode.GRAPH_ONLY,
        run_graph=True,
        run_weather=False,
        run_planning=False,
        reason="knowledge_query",
    )


class TravelOrchestrator:
    def __init__(
        self,
        kb_client: SupportsKbSearch,
        weather_client: OpenMeteoWeatherClient | None,
        planning_agent: SupportsItineraryPlanning | None = None,
        route_client: SupportsRoutes | None = None,
    ) -> None:
        self.kb_client = kb_client
        self.weather_client = weather_client
        self.planning_agent = planning_agent
        self.route_client = route_client

    def run(
        self,
        *,
        message: str,
        session_id: str,
        city: str | None,
        entity_types: list[str] | None,
        top_k: int,
        kb_version: KbVersion,
        travel_date: date | None,
        include_weather: bool | None,
        latitude: float | None,
        longitude: float | None,
        conversation_context: dict[str, Any] | None = None,
    ) -> OrchestratedResult:
        plan = build_orchestration_plan(
            message=message,
            travel_date=travel_date,
            include_weather=include_weather,
            entity_types=entity_types,
        )
        trace = [plan.trace_event()]
        effective_city = city or supported_city_from_coordinates(latitude, longitude)
        weather: WeatherAssessment | None = None
        weather_forecast: list[WeatherAssessment] = []
        weather_trace: dict[str, Any] = {"node": "weather", "status": "skipped"}
        planning_trace: dict[str, Any] = {"node": "planning", "status": "skipped"}
        route_trace: dict[str, Any] = {"node": "route", "status": "skipped"}

        if plan.run_graph and plan.run_weather:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="nextrip-tools") as pool:
                graph_future = pool.submit(
                    self._run_graph,
                    message=message,
                    session_id=session_id,
                    city=effective_city,
                    entity_types=entity_types,
                    top_k=top_k,
                    kb_version=kb_version,
                    conversation_context=conversation_context,
                )
                weather_future = pool.submit(
                    self._run_weather_forecast,
                    message=message,
                    city=effective_city,
                    travel_date=travel_date,
                    include_weather=include_weather,
                    latitude=latitude,
                    longitude=longitude,
                    required_tools=["weather"],
                    duration_days=(
                        requested_itinerary_duration_days(message)
                        if plan.run_planning
                        else 1
                    ),
                )
                graph = graph_future.result()
                weather_forecast, weather_trace = weather_future.result()
                weather = weather_forecast[0] if weather_forecast else None
        elif plan.run_graph:
            graph = self._run_graph(
                message=message,
                session_id=session_id,
                city=effective_city,
                entity_types=entity_types,
                top_k=top_k,
                kb_version=kb_version,
                conversation_context=conversation_context,
            )
        elif plan.run_weather:
            graph = AgentResult(
                answer="",
                answer_type="weather",
                required_tools=["weather"],
            )
            weather_forecast, weather_trace = self._run_weather_forecast(
                message=message,
                city=effective_city,
                travel_date=travel_date,
                include_weather=include_weather,
                latitude=latitude,
                longitude=longitude,
                required_tools=["weather"],
                duration_days=1,
            )
            weather = weather_forecast[0] if weather_forecast else None
        else:
            graph = AgentResult(
                answer=GREETING_ANSWER,
                answer_type="conversation",
            )

        if (
            plan.run_graph
            and not plan.run_weather
            and bool({"weather", "weather_forecast"} & set(graph.required_tools))
        ):
            weather_forecast, weather_trace = self._run_weather_forecast(
                message=message,
                city=effective_city,
                travel_date=travel_date,
                include_weather=include_weather,
                latitude=latitude,
                longitude=longitude,
                required_tools=graph.required_tools,
                duration_days=1,
            )
            weather = weather_forecast[0] if weather_forecast else None
            if weather_trace.get("status") != "skipped":
                plan = OrchestrationPlan(
                    mode=OrchestrationMode.GRAPH_AND_WEATHER,
                    run_graph=True,
                    run_weather=True,
                    run_planning=plan.run_planning,
                    reason="knowledge_agent_requested_weather",
                )
                trace.append(plan.trace_event())

        if weather_trace.get("status") == "needs_input" and "city" not in graph.missing_fields:
            graph = graph.model_copy(
                update={"missing_fields": [*graph.missing_fields, "city"]}
            )

        if plan.run_planning:
            graph, planning_trace = planning_agent_node(
                message=message,
                graph=graph,
                weather_forecast=weather_forecast,
                planner=self.planning_agent,
                city=effective_city,
                latitude=latitude,
                longitude=longitude,
                personalization_context=dict(
                    (conversation_context or {}).get("personalization") or {}
                ),
            )

        if kb_version == "v8" and "route" in graph.required_tools:
            graph, route_trace = self._run_route(
                graph,
                latitude=latitude,
                longitude=longitude,
            )
            if route_trace.get("status") == "completed":
                plan = OrchestrationPlan(
                    mode=OrchestrationMode.GRAPH_AND_ROUTE,
                    run_graph=True,
                    run_weather=plan.run_weather,
                    run_planning=plan.run_planning,
                    reason="v8_actual_route_requested",
                )
                trace.append(plan.trace_event())

        return OrchestratedResult(
            plan=plan,
            graph=graph,
            weather=weather,
            weather_forecast=weather_forecast,
            weather_trace=weather_trace,
            planning_trace=planning_trace,
            route_trace=route_trace,
            trace=trace,
        )

    def _run_route(
        self,
        graph: AgentResult,
        *,
        latitude: float | None,
        longitude: float | None,
    ) -> tuple[AgentResult, dict[str, Any]]:
        context = graph.route_context or {}
        raw_endpoints = list(context.get("endpoints") or [])
        endpoints = [
            RouteWaypoint(
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                place_id=str(item.get("place_id") or "") or None,
                name=str(item.get("name") or "") or None,
            )
            for item in raw_endpoints
            if item.get("latitude") is not None and item.get("longitude") is not None
        ]
        if len(endpoints) >= 2:
            origin, destination = endpoints[:2]
        elif len(endpoints) == 1 and latitude is not None and longitude is not None:
            origin = RouteWaypoint(
                latitude=latitude,
                longitude=longitude,
                name="Vị trí hiện tại",
            )
            destination = endpoints[0]
        else:
            missing = "current_location" if len(endpoints) == 1 else "route_endpoints"
            return graph.model_copy(
                update={
                    "missing_fields": list(
                        dict.fromkeys([*graph.missing_fields, missing])
                    )
                }
            ), {
                "node": "route",
                "status": "needs_input",
                "reason": missing,
            }
        if self.route_client is None or not self.route_client.configured:
            return graph, {
                "node": "route",
                "status": "unavailable",
                "reason": "Google Routes API is not configured.",
            }
        options = dict(context.get("options") or {})
        travel_mode = str(options.get("travel_mode") or "car")
        raw_speed = options.get("speed_kmh")
        speed_kmh = float(raw_speed) if raw_speed is not None else None
        started_at = perf_counter()
        try:
            route = self.route_client.compute_route(
                origin,
                destination,
                travel_mode=travel_mode,
                speed_kmh=speed_kmh,
            )
        except (RouteUnavailable, ValueError) as exc:
            return graph, {
                "node": "route",
                "status": "unavailable",
                "reason": str(exc),
                "elapsed_ms": int((perf_counter() - started_at) * 1000),
            }
        subject_id = origin.place_id or "current-location"
        distance_km = round(route.distance_meters / 1000, 3)
        duration_minutes = max(1, round(route.duration_seconds / 60))
        route_facts = [
            {
                "fact_id": f"route-distance:{subject_id}:{destination.place_id or 'destination'}:{travel_mode}",
                "subject_id": subject_id,
                "predicate": "actual_route_distance",
                "value": distance_km,
                "value_type": "number",
                "unit": "km",
                "confidence": 1.0,
                "evidence_ids": [],
            },
            {
                "fact_id": f"route-duration:{subject_id}:{destination.place_id or 'destination'}:{travel_mode}",
                "subject_id": subject_id,
                "predicate": "route_duration",
                "value": duration_minutes,
                "value_type": "number",
                "unit": "minutes",
                "confidence": 1.0,
                "evidence_ids": [],
            },
        ]
        if route.speed_kmh is not None:
            route_facts.append(
                {
                    "fact_id": f"route-speed:{subject_id}:{destination.place_id or 'destination'}",
                    "subject_id": subject_id,
                    "predicate": "user_speed",
                    "value": route.speed_kmh,
                    "value_type": "number",
                    "unit": "km/h",
                    "confidence": 1.0,
                    "evidence_ids": [],
                }
            )
        trace = {
            "node": "route",
            "status": "completed",
            "provider": route.provider,
            "travel_mode": travel_mode,
            "distance_meters": route.distance_meters,
            "duration_source": route.duration_source,
            "elapsed_ms": int((perf_counter() - started_at) * 1000),
        }
        return graph.model_copy(
            update={
                "answer_type": "route",
                "facts": [*graph.facts, *route_facts],
                "required_tools": [
                    tool for tool in graph.required_tools if tool != "route"
                ],
                "trace": [*graph.trace, trace],
            }
        ), trace
    def _run_graph(
        self,
        *,
        message: str,
        session_id: str,
        city: str | None,
        entity_types: list[str] | None,
        top_k: int,
        kb_version: KbVersion,
        conversation_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        return run_nextrip_agent(
            message=message,
            session_id=session_id,
            city=city,
            entity_types=entity_types,
            top_k=top_k,
            kb_client=self.kb_client,
            kb_version=kb_version,
            answer_generator=None,
            conversation_context=conversation_context,
        )

    def _run_weather_forecast(
        self,
        *,
        message: str,
        city: str | None,
        travel_date: date | None,
        include_weather: bool | None,
        latitude: float | None,
        longitude: float | None,
        required_tools: list[str],
        duration_days: int,
    ) -> tuple[list[WeatherAssessment], dict[str, Any]]:
        should_run = WeatherAgent.should_run(
            message=message,
            travel_date=travel_date,
            include_weather=include_weather,
            required_tools=required_tools,
        )
        if not should_run:
            return [], {"node": "weather", "status": "skipped"}
        if self.weather_client is None or not self.weather_client.configured:
            return [], {
                "node": "weather",
                "status": "unavailable",
                "reason": "Weather client is not configured.",
            }
        started_at = perf_counter()
        try:
            weather = WeatherAgent(self.weather_client).run_range(
                message=message,
                city=city,
                travel_date=travel_date,
                latitude=latitude,
                longitude=longitude,
                duration_days=duration_days,
            )
        except WeatherLocationRequired as exc:
            return [], {
                "node": "weather",
                "status": "needs_input",
                "code": "missing_location",
                "reason": str(exc),
                "elapsed_ms": int((perf_counter() - started_at) * 1000),
            }
        except WeatherUnavailable as exc:
            return [], {
                "node": "weather",
                "status": "unavailable",
                "reason": str(exc),
                "elapsed_ms": int((perf_counter() - started_at) * 1000),
            }
        return weather, {
            "node": "weather",
            "status": "completed",
            "forecast_days": len(weather),
            "suitability": [item.suitability for item in weather],
            "elapsed_ms": int((perf_counter() - started_at) * 1000),
        }


def _is_greeting(message: str) -> bool:
    normalized = " ".join(normalize_text(message).strip().strip("!.,?").split())
    return normalized in GREETING_MESSAGES
