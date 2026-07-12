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
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.weather import (
    WEATHER_TERMS,
    WeatherAgent,
    WeatherAssessment,
    normalize_text,
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


class OrchestrationMode(str, Enum):
    GRAPH_ONLY = "graph_only"
    WEATHER_ONLY = "weather_only"
    GRAPH_AND_WEATHER = "graph_and_weather"


@dataclass(frozen=True)
class OrchestrationPlan:
    mode: OrchestrationMode
    run_graph: bool
    run_weather: bool
    reason: str

    def trace_event(self) -> dict[str, Any]:
        return {
            "node": "orchestrator",
            "status": "planned",
            "mode": self.mode.value,
            "run_graph": self.run_graph,
            "run_weather": self.run_weather,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OrchestratedResult:
    plan: OrchestrationPlan
    graph: AgentResult
    weather: WeatherAssessment | None
    weather_trace: dict[str, Any]
    trace: list[dict[str, Any]]


def build_orchestration_plan(
    *,
    message: str,
    travel_date: date | None,
    include_weather: bool | None,
    entity_types: list[str] | None,
) -> OrchestrationPlan:
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
            reason="explicit_weather_query",
        )
    if run_weather:
        return OrchestrationPlan(
            mode=OrchestrationMode.GRAPH_AND_WEATHER,
            run_graph=True,
            run_weather=True,
            reason="travel_query_with_weather_context",
        )
    return OrchestrationPlan(
        mode=OrchestrationMode.GRAPH_ONLY,
        run_graph=True,
        run_weather=False,
        reason="knowledge_query",
    )


class TravelOrchestrator:
    def __init__(
        self,
        kb_client: SupportsKbSearch,
        weather_client: OpenMeteoWeatherClient | None,
    ) -> None:
        self.kb_client = kb_client
        self.weather_client = weather_client

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
    ) -> OrchestratedResult:
        plan = build_orchestration_plan(
            message=message,
            travel_date=travel_date,
            include_weather=include_weather,
            entity_types=entity_types,
        )
        trace = [plan.trace_event()]
        weather: WeatherAssessment | None = None
        weather_trace: dict[str, Any] = {"node": "weather", "status": "skipped"}

        if plan.run_graph and plan.run_weather:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="nextrip-tools") as pool:
                graph_future = pool.submit(
                    self._run_graph,
                    message=message,
                    session_id=session_id,
                    city=city,
                    entity_types=entity_types,
                    top_k=top_k,
                    kb_version=kb_version,
                )
                weather_future = pool.submit(
                    self._run_weather,
                    message=message,
                    city=city,
                    travel_date=travel_date,
                    include_weather=include_weather,
                    latitude=latitude,
                    longitude=longitude,
                    required_tools=["weather"],
                )
                graph = graph_future.result()
                weather, weather_trace = weather_future.result()
        elif plan.run_graph:
            graph = self._run_graph(
                message=message,
                session_id=session_id,
                city=city,
                entity_types=entity_types,
                top_k=top_k,
                kb_version=kb_version,
            )
        else:
            graph = AgentResult(
                answer="",
                answer_type="weather",
                required_tools=["weather"],
            )
            weather, weather_trace = self._run_weather(
                message=message,
                city=city,
                travel_date=travel_date,
                include_weather=include_weather,
                latitude=latitude,
                longitude=longitude,
                required_tools=["weather"],
            )

        if (
            plan.run_graph
            and not plan.run_weather
            and "weather" in graph.required_tools
        ):
            weather, weather_trace = self._run_weather(
                message=message,
                city=city,
                travel_date=travel_date,
                include_weather=include_weather,
                latitude=latitude,
                longitude=longitude,
                required_tools=graph.required_tools,
            )
            if weather_trace.get("status") != "skipped":
                plan = OrchestrationPlan(
                    mode=OrchestrationMode.GRAPH_AND_WEATHER,
                    run_graph=True,
                    run_weather=True,
                    reason="knowledge_agent_requested_weather",
                )
                trace.append(plan.trace_event())

        if weather_trace.get("status") == "needs_input" and "city" not in graph.missing_fields:
            graph = graph.model_copy(
                update={"missing_fields": [*graph.missing_fields, "city"]}
            )

        return OrchestratedResult(
            plan=plan,
            graph=graph,
            weather=weather,
            weather_trace=weather_trace,
            trace=trace,
        )

    def _run_graph(
        self,
        *,
        message: str,
        session_id: str,
        city: str | None,
        entity_types: list[str] | None,
        top_k: int,
        kb_version: KbVersion,
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
        )

    def _run_weather(
        self,
        *,
        message: str,
        city: str | None,
        travel_date: date | None,
        include_weather: bool | None,
        latitude: float | None,
        longitude: float | None,
        required_tools: list[str],
    ) -> tuple[WeatherAssessment | None, dict[str, Any]]:
        should_run = WeatherAgent.should_run(
            message=message,
            travel_date=travel_date,
            include_weather=include_weather,
            required_tools=required_tools,
        )
        if not should_run:
            return None, {"node": "weather", "status": "skipped"}
        if self.weather_client is None or not self.weather_client.configured:
            return None, {
                "node": "weather",
                "status": "unavailable",
                "reason": "Weather client is not configured.",
            }
        started_at = perf_counter()
        try:
            weather = WeatherAgent(self.weather_client).run(
                message=message,
                city=city,
                travel_date=travel_date,
                latitude=latitude,
                longitude=longitude,
            )
        except WeatherLocationRequired as exc:
            return None, {
                "node": "weather",
                "status": "needs_input",
                "code": "missing_location",
                "reason": str(exc),
                "elapsed_ms": int((perf_counter() - started_at) * 1000),
            }
        except WeatherUnavailable as exc:
            return None, {
                "node": "weather",
                "status": "unavailable",
                "reason": str(exc),
                "elapsed_ms": int((perf_counter() - started_at) * 1000),
            }
        return weather, {
            "node": "weather",
            "status": "completed",
            "suitability": weather.suitability,
            "elapsed_ms": int((perf_counter() - started_at) * 1000),
        }
