from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from enum import Enum
from time import perf_counter
from typing import Any

from src.core_ai.nextrip_agent.constants import KbVersion
from src.core_ai.nextrip_agent.current_data import (
    SupportsCurrentData,
    enrich_current_data,
)
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.core_ai.nextrip_agent.nodes.knowledge import SupportsKbSearch
from src.core_ai.nextrip_agent.planning import (
    DEFAULT_PLANNING_POLICY,
    PlanningPolicy,
    SupportsItineraryPlanning,
    is_itinerary_request,
    planning_agent_node,
    requested_itinerary_duration_days,
    requested_itinerary_duration_nights,
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
    trace: list[dict[str, Any]]


@dataclass(frozen=True)
class _CandidateCoverageRequirement:
    name: str
    accepted_entity_types: frozenset[str]
    query_entity_type: str
    minimum: int
    query_label: str


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
        current_data_client: SupportsCurrentData | None = None,
        planning_policy: PlanningPolicy = DEFAULT_PLANNING_POLICY,
    ) -> None:
        self.kb_client = kb_client
        self.weather_client = weather_client
        self.planning_agent = planning_agent
        self.current_data_client = current_data_client
        self.planning_policy = planning_policy

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
        is_v8 = str(kb_version).lower() == "v8"
        graph_top_k = max(top_k, 20) if plan.run_planning and is_v8 else top_k
        weather: WeatherAssessment | None = None
        weather_forecast: list[WeatherAssessment] = []
        weather_trace: dict[str, Any] = {"node": "weather", "status": "skipped"}
        planning_trace: dict[str, Any] = {"node": "planning", "status": "skipped"}

        if plan.run_graph and plan.run_weather:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="nextrip-tools") as pool:
                graph_future = pool.submit(
                    self._run_graph,
                    message=message,
                    session_id=session_id,
                    city=effective_city,
                    entity_types=entity_types,
                    top_k=graph_top_k,
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
                top_k=graph_top_k,
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
            duration_days = requested_itinerary_duration_days(
                message,
                graph.query_plan,
            )
            duration_nights = requested_itinerary_duration_nights(
                message,
                duration_days,
                graph.query_plan,
            )
            graph = graph.model_copy(
                update={
                    "query_plan": {
                        **graph.query_plan,
                        "duration_days": duration_days,
                        "duration_nights": duration_nights,
                    }
                }
            )
            graph, candidate_coverage_trace = self._supplement_planning_candidates(
                graph,
                message=message,
                session_id=session_id,
                city=effective_city,
                top_k=graph_top_k,
                kb_version=kb_version,
                conversation_context=conversation_context,
                weather_forecast=weather_forecast,
                duration_days=duration_days,
                duration_nights=duration_nights,
            )
            trace.append(candidate_coverage_trace)
            graph, preplanning_current_trace = enrich_current_data(
                graph,
                self.current_data_client,
                travel_date=travel_date,
                include_traffic=False,
            )
            trace.append(
                {
                    **preplanning_current_trace,
                    "node": "current_data_preplanning",
                }
            )
            graph, planning_trace = planning_agent_node(
                message=message,
                graph=graph,
                weather_forecast=weather_forecast,
                # Gemini now returns only grounded semantic assignments. Python
                # owns clock scheduling, hotel lifecycle, and validation, so V8
                # can use the planner without trusting LLM-generated arithmetic.
                planner=self.planning_agent,
                city=effective_city,
                latitude=latitude,
                longitude=longitude,
                personalization_context=dict(
                    (conversation_context or {}).get("personalization") or {}
                ),
                route_provider=self.current_data_client,
                travel_date=travel_date,
                policy=self.planning_policy,
            )

        return OrchestratedResult(
            plan=plan,
            graph=graph,
            weather=weather,
            weather_forecast=weather_forecast,
            weather_trace=weather_trace,
            planning_trace=planning_trace,
            trace=trace,
        )

    def _supplement_planning_candidates(
        self,
        graph: AgentResult,
        *,
        message: str,
        session_id: str,
        city: str | None,
        top_k: int,
        kb_version: KbVersion,
        conversation_context: dict[str, Any] | None,
        weather_forecast: list[WeatherAssessment],
        duration_days: int,
        duration_nights: int,
    ) -> tuple[AgentResult, dict[str, Any]]:
        requirements = _candidate_coverage_requirements(
            duration_days=duration_days,
            duration_nights=duration_nights,
            policy=self.planning_policy,
            weather_forecast=weather_forecast,
        )
        if not requirements or graph.error:
            return graph, {
                "node": "planning_candidate_coverage",
                "status": "skipped" if not requirements else "unavailable",
                "duration_days": duration_days,
                "duration_nights": duration_nights,
            }

        evidence = list(graph.evidence)
        requests: list[dict[str, Any]] = []
        rainy_trip = any(item.suitability == "unsuitable" for item in weather_forecast)
        pending: list[tuple[_CandidateCoverageRequirement, int, str, int]] = []
        for requirement in requirements:
            before = _count_eligible_candidates(
                evidence,
                requirement,
                city=city,
            )
            if before >= requirement.minimum:
                continue
            query = _supplemental_candidate_query(
                requirement,
                message=message,
                indoor_only=rainy_trip and requirement.name == "activity",
            )
            request_top_k = min(
                top_k,
                self.planning_policy.supplemental_candidate_limit,
                max(requirement.minimum - before, requirement.minimum),
            )
            pending.append((requirement, before, query, request_top_k))

        supplemental_results: list[AgentResult] = []
        if pending:
            with ThreadPoolExecutor(
                max_workers=min(4, len(pending)),
                thread_name_prefix="nextrip-coverage",
            ) as pool:
                futures = [
                    pool.submit(
                        self._run_graph,
                        message=query,
                        session_id=f"{session_id}:coverage:{requirement.name}",
                        city=city,
                        entity_types=[requirement.query_entity_type],
                        top_k=request_top_k,
                        kb_version=kb_version,
                        conversation_context=conversation_context,
                    )
                    for requirement, _, query, request_top_k in pending
                ]
                # Consume results in requirement order so evidence and trace
                # remain deterministic even though the I/O is concurrent.
                supplemental_results = [future.result() for future in futures]

        for (requirement, before, _, _), supplemental in zip(
            pending,
            supplemental_results,
            strict=True,
        ):
            accepted = [
                item
                for item in supplemental.evidence
                if _candidate_matches_requirement(
                    item,
                    requirement,
                    city=city,
                )
            ]
            evidence = _merge_candidate_evidence(evidence, accepted)
            after = _count_eligible_candidates(
                evidence,
                requirement,
                city=city,
            )
            requests.append(
                {
                    "requirement": requirement.name,
                    "entity_type": requirement.query_entity_type,
                    "minimum": requirement.minimum,
                    "before": before,
                    "accepted": len(accepted),
                    "after": after,
                    "status": "completed" if after >= requirement.minimum else "partial",
                    "error": supplemental.error,
                }
            )

        missing = [
            requirement.name
            for requirement in requirements
            if _count_eligible_candidates(evidence, requirement, city=city)
            < requirement.minimum
        ]
        warnings = list(graph.warnings)
        warnings.extend(
            f"planning_candidate_coverage_missing:{name}" for name in missing
        )
        updated = graph.model_copy(
            update={
                "evidence": evidence,
                "warnings": list(dict.fromkeys(warnings)),
            }
        )
        return updated, {
            "node": "planning_candidate_coverage",
            "status": "completed" if not missing else "partial",
            "duration_days": duration_days,
            "duration_nights": duration_nights,
            "requests": requests,
            "missing": missing,
            "candidate_count": len(evidence),
        }

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


def _candidate_coverage_requirements(
    *,
    duration_days: int,
    duration_nights: int,
    policy: PlanningPolicy,
    weather_forecast: list[WeatherAssessment],
) -> list[_CandidateCoverageRequirement]:
    full_trip = (
        duration_nights > 0
        or duration_days >= policy.full_coverage_min_days
    )
    if not full_trip:
        return []
    unsuitable_days = sum(
        item.suitability == "unsuitable"
        for item in weather_forecast[:duration_days]
    )
    activity_days = (
        duration_days
        if not weather_forecast
        else max(1, duration_days - unsuitable_days)
    )
    requirements = [
        _CandidateCoverageRequirement(
            name="activity",
            accepted_entity_types=frozenset({"attraction", "nightlife"}),
            query_entity_type="attraction",
            minimum=activity_days * policy.activity_candidates_per_day,
            query_label="địa điểm tham quan",
        ),
        _CandidateCoverageRequirement(
            name="meal",
            accepted_entity_types=frozenset({"restaurant"}),
            query_entity_type="restaurant",
            minimum=duration_days * policy.meal_candidates_per_day,
            query_label="nhà hàng hoặc quán ăn",
        ),
    ]
    if policy.rest_break_candidates_per_trip:
        requirements.append(
            _CandidateCoverageRequirement(
                name="rest_break",
                accepted_entity_types=frozenset({"cafe"}),
                query_entity_type="cafe",
                minimum=policy.rest_break_candidates_per_trip,
                query_label="quán cà phê để nghỉ ngơi",
            )
        )
    if duration_nights > 0:
        requirements.append(
            _CandidateCoverageRequirement(
                name="hotel",
                accepted_entity_types=frozenset({"hotel"}),
                query_entity_type="hotel",
                minimum=policy.hotel_candidate_options,
                query_label="khách sạn phù hợp để lưu trú",
            )
        )
    return requirements


def _supplemental_candidate_query(
    requirement: _CandidateCoverageRequirement,
    *,
    message: str,
    indoor_only: bool,
) -> str:
    label = requirement.query_label
    if indoor_only:
        label = f"{label} trong nhà"
    return f"Bổ sung {label} đã xác minh cho yêu cầu: {message}"


def _count_eligible_candidates(
    evidence: list[dict[str, Any]],
    requirement: _CandidateCoverageRequirement,
    *,
    city: str | None,
) -> int:
    return sum(
        _candidate_matches_requirement(item, requirement, city=city)
        for item in evidence
    )


def _candidate_matches_requirement(
    candidate: dict[str, Any],
    requirement: _CandidateCoverageRequirement,
    *,
    city: str | None,
) -> bool:
    if candidate.get("entity_type") not in requirement.accepted_entity_types:
        return False
    if not candidate.get("place_id"):
        return False
    return city is None or normalize_text(str(candidate.get("city") or "")) == normalize_text(
        city
    )


def _merge_candidate_evidence(
    primary: list[dict[str, Any]],
    supplemental: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(item) for item in primary]
    index_by_id = {
        str(item.get("place_id") or ""): index
        for index, item in enumerate(merged)
        if item.get("place_id")
    }
    for item in supplemental:
        place_id = str(item.get("place_id") or "")
        if not place_id:
            continue
        existing_index = index_by_id.get(place_id)
        if existing_index is None:
            index_by_id[place_id] = len(merged)
            merged.append(dict(item))
            continue
        existing = merged[existing_index]
        existing_attributes = existing.get("attributes")
        incoming_attributes = item.get("attributes")
        attributes = {
            **(
                dict(existing_attributes)
                if isinstance(existing_attributes, dict)
                else {}
            ),
            **(
                dict(incoming_attributes)
                if isinstance(incoming_attributes, dict)
                else {}
            ),
        }
        merged[existing_index] = {
            **existing,
            **{key: value for key, value in item.items() if value is not None},
            **({"attributes": attributes} if attributes else {}),
        }
    return merged


def _is_greeting(message: str) -> bool:
    normalized = " ".join(normalize_text(message).strip().strip("!.,?").split())
    return normalized in GREETING_MESSAGES
