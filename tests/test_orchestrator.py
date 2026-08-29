from __future__ import annotations

from datetime import date
from threading import Barrier, Lock, get_ident

import pytest

from src.core_ai.nextrip_agent.orchestrator import (
    OrchestrationMode,
    TravelOrchestrator,
    build_orchestration_plan,
)
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.synthesizer import (
    AnswerGenerationUnavailableError,
    synthesize_answer,
)
from src.core_ai.nextrip_agent.weather import WeatherAssessment
from src.infra.weather import DailyForecast
from src.shared.request_context import current_request_id, reset_request_id, set_request_id


class FakeKbClient:
    def __init__(self) -> None:
        self.calls = 0

    def query_typed(self, *, query, top_k, kb_version="v4"):
        self.calls += 1
        return {
            "kb_version": kb_version,
            "answer_type": "recommendation",
            "recommendations": [{
                "place_id": "attr_dn_001",
                "name": "Bãi biển Mỹ Khê",
                "city": "Đà Nẵng",
                "entity_type": "attraction",
                "category": "Biển đảo",
            }],
            "facts": [],
            "evidence": [{
                "subject_id": "attr_dn_001",
                "source_name": "verified-source",
                "url": "https://example.test/my-khe",
            }],
            "missing_fields": [],
            "query_plan": {"retrieval_mode": "recommendation"},
            "matched_paths": [],
            "constraint_results": [],
            "required_tools": [],
            "trace": [{"step": "retrieval", "status": "ok"}],
        }


class FailingKbClient:
    def query_typed(self, **kwargs):
        raise AssertionError("Weather-only route must not call GraphRAG")


class CandidateCoverageKbClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query_typed(
        self,
        *,
        query,
        top_k,
        kb_version="v8",
        conversation_context=None,
    ):
        self.calls.append({"query": query, "top_k": top_k})
        normalized = query.casefold()
        if "khách sạn" in normalized:
            entity_type = "hotel"
            prefix = "hotel"
        elif "nhà hàng" in normalized:
            entity_type = "restaurant"
            prefix = "rest"
        elif "tham quan" in normalized:
            entity_type = "attraction"
            prefix = "attr"
        else:
            entity_type = "cafe"
            prefix = "cafe"
        return {
            "kb_version": kb_version,
            "answer_type": "recommendation",
            "recommendations": [
                {
                    "place_id": f"{prefix}_qn_{index:03d}",
                    "name": f"{entity_type} {index}",
                    "city": "Quy Nhơn",
                    "entity_type": entity_type,
                    "category": entity_type,
                    "attributes": {
                        "opening_hours_open": "06:00",
                        "opening_hours_close": "23:00",
                        **({"indoor": True} if entity_type == "attraction" else {}),
                    },
                }
                for index in range(1, top_k + 1)
            ],
            "facts": [],
            "evidence": [],
            "missing_fields": [],
            "query_plan": {"intent": "plan_candidates", "duration_days": 3},
            "matched_paths": [],
            "constraint_results": [],
            "required_tools": [],
            "trace": [],
        }


class ParallelCandidateCoverageKbClient(CandidateCoverageKbClient):
    def __init__(self) -> None:
        super().__init__()
        self._call_lock = Lock()
        self._call_number = 0
        self._supplement_barrier = Barrier(3, timeout=2)
        self.supplement_threads: set[int] = set()
        self.request_ids: list[str] = []

    def query_typed(self, **kwargs):
        with self._call_lock:
            self._call_number += 1
            call_number = self._call_number
            self.request_ids.append(current_request_id())
        if call_number > 1:
            self.supplement_threads.add(get_ident())
            self._supplement_barrier.wait()
        return super().query_typed(**kwargs)


class DurationlessCoverageKbClient(CandidateCoverageKbClient):
    def query_typed(self, **kwargs):
        result = super().query_typed(**kwargs)
        result["query_plan"] = {"intent": "plan_candidates"}
        return result


class StaleHotelCoverageKbClient(CandidateCoverageKbClient):
    def query_typed(self, **kwargs):
        result = super().query_typed(**kwargs)
        for item in result["recommendations"]:
            if item["entity_type"] != "hotel":
                continue
            item["attributes"]["hotel_availability"] = {
                "selected_window_index": None,
                "windows": [
                    {
                        "requested_check_in": "2026-08-27",
                        "stay_nights": 1,
                        "lookup_status": "available",
                        "availability": "unavailable",
                        "offers": [],
                    }
                ],
            }
        return result


class CapturingCurrentData:
    def __init__(self) -> None:
        self.hotel_requests: list[dict] = []

    def places(self, place_ids):
        return {"items": []}

    def hotel_availability(self, **kwargs):
        self.hotel_requests.append(kwargs)
        return {"results": []}

    def recommend_transport(self, **kwargs):
        return {
            "status": "recommended",
            "recommended_mode": "drive",
            "options": [
                {
                    "recommended": True,
                    "distance_meters": 1000,
                    "duration_seconds": 600,
                    "route": {"route": {"provider": "here"}},
                }
            ],
        }


class FakeWeatherClient:
    configured = True

    def __init__(self) -> None:
        self.calls = 0

    def forecast(self, latitude, longitude, target_date, today):
        self.calls += 1
        return DailyForecast(
            forecast_date=target_date,
            condition="Có mây",
            condition_type="PARTLY_CLOUDY",
            min_temperature_c=25,
            max_temperature_c=31,
            precipitation_probability=20,
            thunderstorm_probability=5,
            uv_index=6,
            wind_gust_kph=18,
        )

    def forecast_range(self, latitude, longitude, start_date, duration_days, today):
        return [
            self.forecast(latitude, longitude, start_date, today)
            for _ in range(duration_days)
        ]


class ContextCapturingKbClient(FakeKbClient):
    def __init__(self) -> None:
        super().__init__()
        self.request_ids: list[str] = []

    def query_typed(self, **kwargs):
        self.request_ids.append(current_request_id())
        return super().query_typed(**kwargs)


class ContextCapturingWeatherClient(FakeWeatherClient):
    def __init__(self) -> None:
        super().__init__()
        self.request_ids: list[str] = []

    def forecast_range(self, latitude, longitude, start_date, duration_days, today):
        self.request_ids.append(current_request_id())
        return super().forecast_range(
            latitude,
            longitude,
            start_date,
            duration_days,
            today,
        )


class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        raise AssertionError("Combined flow must use synthesize()")

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return "Mỹ Khê phù hợp tham quan; thời tiết có mây và ít khả năng mưa."


class FailingSynthesizer:
    def generate(self, **kwargs):
        raise RuntimeError("Gemini unavailable")

    def synthesize(self, **kwargs):
        raise RuntimeError("Gemini unavailable")


def test_orchestrator_plans_graph_only() -> None:
    plan = build_orchestration_plan(
        message="Gợi ý quán ăn gia đình tại Quy Nhơn",
        travel_date=None,
        include_weather=None,
        entity_types=["restaurant"],
    )

    assert plan.mode == OrchestrationMode.GRAPH_ONLY
    assert plan.run_graph is True
    assert plan.run_weather is False


def test_orchestrator_routes_itinerary_through_graph_weather_and_planning() -> None:
    plan = build_orchestration_plan(
        message="Tôi ở Quy Nhơn 2 ngày 1 đêm, lộ trình thế nào?",
        travel_date=None,
        include_weather=None,
        entity_types=None,
    )

    assert plan.mode == OrchestrationMode.ITINERARY_PLANNING
    assert plan.run_graph is True
    assert plan.run_weather is True
    assert plan.run_planning is True


def test_orchestrator_propagates_request_context_to_primary_tool_threads() -> None:
    kb_client = ContextCapturingKbClient()
    weather_client = ContextCapturingWeatherClient()
    token = set_request_id("request-context-primary-tools")
    try:
        TravelOrchestrator(kb_client, weather_client).run(
            message="Gá»£i Ã½ Ä‘á»‹a Ä‘iá»ƒm vÃ  xem thá»i tiáº¿t ÄÃ  Náºµng",
            session_id="context-primary-tools",
            city="ÄÃ  Náºµng",
            entity_types=None,
            top_k=5,
            kb_version="v4",
            travel_date=date(2026, 8, 28),
            include_weather=True,
            latitude=16.05,
            longitude=108.2,
        )
    finally:
        reset_request_id(token)

    assert kb_client.request_ids == ["request-context-primary-tools"]
    assert weather_client.request_ids == ["request-context-primary-tools"]


def test_orchestrator_supplements_generic_results_with_required_entity_coverage() -> (
    None
):
    kb_client = CandidateCoverageKbClient()

    result = TravelOrchestrator(kb_client, None).run(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        session_id="balanced-planning-candidates",
        city="Quy Nhơn",
        entity_types=None,
        top_k=5,
        kb_version="v8",
        travel_date=None,
        include_weather=False,
        latitude=None,
        longitude=None,
    )

    coverage = next(
        item for item in result.trace if item["node"] == "planning_candidate_coverage"
    )
    assert coverage["status"] == "completed"
    assert {item["requirement"] for item in coverage["requests"]} == {
        "activity",
        "meal",
        "hotel",
    }
    assert coverage["missing"] == []
    assert result.planning_trace["status"] == "completed"
    assert all(
        any(slot["role"] == "activity" for slot in day["slots"])
        and any(slot["role"] == "meal" for slot in day["slots"])
        for day in result.graph.itinerary
    )
    hotel_slots = [
        (day["day"], slot["role"], slot["place_id"])
        for day in result.graph.itinerary
        for slot in day["slots"]
        if slot["entity_type"] == "hotel"
    ]
    assert hotel_slots == [
        (1, "check_in", "hotel_qn_001"),
        (3, "check_out", "hotel_qn_001"),
    ]


def test_orchestrator_runs_typed_candidate_supplements_in_parallel() -> None:
    kb_client = ParallelCandidateCoverageKbClient()

    result = TravelOrchestrator(kb_client, None).run(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        session_id="parallel-planning-candidates",
        city="Quy Nhơn",
        entity_types=None,
        top_k=5,
        kb_version="v8",
        travel_date=None,
        include_weather=False,
        latitude=None,
        longitude=None,
    )

    coverage = next(
        item for item in result.trace if item["node"] == "planning_candidate_coverage"
    )
    assert coverage["status"] == "completed"
    assert [item["requirement"] for item in coverage["requests"]] == [
        "activity",
        "meal",
        "hotel",
    ]
    assert len(kb_client.supplement_threads) == 3
    entity_order = list(
        dict.fromkeys(item["entity_type"] for item in result.graph.evidence)
    )
    assert entity_order == ["cafe", "attraction", "restaurant", "hotel"]


def test_orchestrator_propagates_request_context_to_candidate_threads() -> None:
    kb_client = ParallelCandidateCoverageKbClient()
    token = set_request_id("request-context-coverage")
    try:
        TravelOrchestrator(kb_client, None).run(
            message="LÃªn lá»‹ch trÃ¬nh Quy NhÆ¡n 3 ngÃ y 2 Ä‘Ãªm",
            session_id="context-planning-candidates",
            city="Quy NhÆ¡n",
            entity_types=None,
            top_k=5,
            kb_version="v8",
            travel_date=None,
            include_weather=False,
            latitude=None,
            longitude=None,
        )
    finally:
        reset_request_id(token)

    assert kb_client.request_ids
    assert set(kb_client.request_ids) == {"request-context-coverage"}


def test_orchestrator_resolves_trip_duration_before_current_data() -> None:
    current_data = CapturingCurrentData()

    result = TravelOrchestrator(
        DurationlessCoverageKbClient(),
        None,
        current_data_client=current_data,
    ).run(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        session_id="resolved-planning-duration",
        city="Quy Nhơn",
        entity_types=None,
        top_k=5,
        kb_version="v8",
        travel_date=date(2026, 8, 28),
        include_weather=False,
        latitude=None,
        longitude=None,
    )

    assert result.graph.query_plan["duration_days"] == 3
    assert result.graph.query_plan["duration_nights"] == 2
    assert current_data.hotel_requests
    assert current_data.hotel_requests[0]["stay_nights"] == 2


def test_orchestrator_replaces_stale_kb_hotel_context_before_planning() -> None:
    current_data = CapturingCurrentData()

    result = TravelOrchestrator(
        StaleHotelCoverageKbClient(),
        None,
        current_data_client=current_data,
    ).run(
        message="Len lich trinh Quy Nhon 3 ngay 2 dem",
        session_id="stale-hotel-context",
        city="Quy Nhon",
        entity_types=None,
        top_k=5,
        kb_version="v8",
        travel_date=date(2026, 8, 28),
        include_weather=False,
        latitude=None,
        longitude=None,
    )

    assert current_data.hotel_requests
    assert current_data.hotel_requests[0]["stay_nights"] == 2
    assert result.planning_trace["status"] == "completed"
    assert len(result.graph.itinerary) == 3


@pytest.mark.parametrize("message", ["hello", "Xin chào!", "Hi NexTrip"])
def test_orchestrator_routes_greetings_without_graph_or_weather(message) -> None:
    plan = build_orchestration_plan(
        message=message,
        travel_date=None,
        include_weather=None,
        entity_types=None,
    )

    assert plan.mode == OrchestrationMode.CONVERSATION
    assert plan.run_graph is False
    assert plan.run_weather is False
    assert plan.reason == "greeting"


def test_greeting_returns_helpful_answer_without_calling_graph() -> None:
    result = TravelOrchestrator(FailingKbClient(), None).run(
        message="hello",
        session_id="greeting",
        city=None,
        entity_types=None,
        top_k=5,
        kb_version="v5",
        travel_date=None,
        include_weather=None,
        latitude=None,
        longitude=None,
    )

    assert result.plan.mode == OrchestrationMode.CONVERSATION
    assert "Quy Nhơn" in result.graph.answer
    assert "Đà Nẵng" in result.graph.answer


def test_orchestrator_weather_only_skips_graph() -> None:
    weather_client = FakeWeatherClient()
    result = TravelOrchestrator(FailingKbClient(), weather_client).run(
        message="Thời tiết Đà Nẵng hôm nay thế nào?",
        session_id="weather-only",
        city="Đà Nẵng",
        entity_types=None,
        top_k=5,
        kb_version="v4",
        travel_date=date(2026, 7, 12),
        include_weather=None,
        latitude=None,
        longitude=None,
    )

    assert result.plan.mode == OrchestrationMode.WEATHER_ONLY
    assert result.graph.evidence == []
    assert result.weather is not None
    assert weather_client.calls == 1


def test_orchestrator_runs_graph_and_weather() -> None:
    kb_client = FakeKbClient()
    weather_client = FakeWeatherClient()
    result = TravelOrchestrator(kb_client, weather_client).run(
        message="Gợi ý địa điểm ở Đà Nẵng, hôm nay trời có mưa không?",
        session_id="combined",
        city="Đà Nẵng",
        entity_types=["attraction"],
        top_k=5,
        kb_version="v4",
        travel_date=date(2026, 7, 12),
        include_weather=None,
        latitude=None,
        longitude=None,
    )

    assert result.plan.mode == OrchestrationMode.GRAPH_AND_WEATHER
    assert result.graph.evidence[0]["place_id"] == "attr_dn_001"
    assert result.weather is not None
    assert kb_client.calls == 1
    assert weather_client.calls == 1


def test_synthesizer_receives_graph_and_weather_context() -> None:
    generator = FakeSynthesizer()
    weather = WeatherAssessment(
        location="Đà Nẵng",
        forecast_date=date(2026, 7, 12),
        condition="Có mây",
        min_temperature_c=25,
        max_temperature_c=31,
        precipitation_probability=20,
        suitability="suitable",
        advice="Phù hợp để tham quan.",
    )
    graph = AgentResult(
        answer="",
        answer_type="recommendation",
        evidence=[{
            "place_id": "attr_dn_001",
            "name": "Bãi biển Mỹ Khê",
            "city": "Đà Nẵng",
            "entity_type": "attraction",
            "category": "Biển đảo",
        }],
    )

    result = synthesize_answer(
        question="Hôm nay nên đi đâu ở Đà Nẵng?",
        kb_version="v4",
        graph=graph,
        graph_used=True,
        weather=weather,
        weather_requested=True,
        weather_trace={"node": "weather", "status": "completed"},
        answer_generator=generator,
    )

    assert result.trace["generator"] == "llm_grounded_combined"
    assert result.trace["sources"] == ["graphrag", "weather"]
    assert generator.calls[0]["weather"]["suitability"] == "suitable"


def test_synthesizer_marks_weather_forecast_tool_as_resolved() -> None:
    generator = FakeSynthesizer()
    graph = AgentResult(
        answer="",
        answer_type="recommendation",
        evidence=[{
            "place_id": "attr_dn_001",
            "name": "Bảo tàng Đà Nẵng",
            "city": "Đà Nẵng",
            "entity_type": "attraction",
        }],
        required_tools=["weather_forecast"],
    )
    weather = WeatherAssessment(
        location="Đà Nẵng",
        forecast_date=date(2026, 7, 15),
        condition="Mưa phùn nhẹ",
        min_temperature_c=29,
        max_temperature_c=35,
        precipitation_probability=21,
        suitability="caution",
        advice="Nên ưu tiên không gian trong nhà.",
    )

    result = synthesize_answer(
        question="Trời mưa thì nên đi đâu ở Đà Nẵng?",
        kb_version="v5",
        graph=graph,
        graph_used=True,
        weather=weather,
        weather_requested=True,
        weather_trace={"node": "weather", "status": "completed"},
        answer_generator=generator,
    )

    assert result.unresolved_tools == []
    assert result.trace["generator"] == "llm_grounded_combined"


def test_synthesizer_requires_configured_gemini_for_graph_context() -> None:
    graph = AgentResult(
        answer="Template answer from Neo4j",
        answer_type="recommendation",
        evidence=[{
            "place_id": "attr_dn_001",
            "name": "Bãi biển Mỹ Khê",
            "city": "Đà Nẵng",
            "entity_type": "attraction",
        }],
    )

    with pytest.raises(
        AnswerGenerationUnavailableError,
        match="Gemini answer generation is temporarily unavailable",
    ):
        synthesize_answer(
            question="Gợi ý địa điểm ở Đà Nẵng",
            kb_version="v5",
            graph=graph,
            graph_used=True,
            weather=None,
            weather_requested=False,
            weather_trace={"node": "weather", "status": "skipped"},
            answer_generator=None,
        )


def test_synthesizer_uses_grounded_fallback_when_gemini_runtime_fails() -> None:
    graph = AgentResult(
        answer="Template answer from Neo4j",
        answer_type="recommendation",
        evidence=[
            {
                "place_id": "attr_dn_001",
                "name": "Bãi biển Mỹ Khê",
                "city": "Đà Nẵng",
                "entity_type": "attraction",
            }
        ],
    )

    result = synthesize_answer(
        question="Gợi ý địa điểm ở Đà Nẵng",
        kb_version="v5",
        graph=graph,
        graph_used=True,
        weather=None,
        weather_requested=False,
        weather_trace={"node": "weather", "status": "skipped"},
        answer_generator=FailingSynthesizer(),
    )

    assert "Bãi biển Mỹ Khê" in result.answer
    assert result.trace["status"] == "fallback"
    assert result.trace["reason"] == "RuntimeError"


def test_synthesizer_reports_unavailable_requested_weather() -> None:
    graph = AgentResult(
        answer="",
        answer_type="recommendation",
        evidence=[{
            "place_id": "attr_dn_001",
            "name": "Bãi biển Mỹ Khê",
            "city": "Đà Nẵng",
            "entity_type": "attraction",
            "category": "Biển đảo",
        }],
    )

    result = synthesize_answer(
        question="Gợi ý địa điểm và xem thời tiết Đà Nẵng",
        kb_version="v4",
        graph=graph,
        graph_used=True,
        weather=None,
        weather_requested=True,
        weather_trace={"node": "weather", "status": "unavailable"},
        answer_generator=FakeSynthesizer(),
    )

    assert "weather" in result.unresolved_tools
    assert "chưa thể lấy dữ liệu thời tiết" in result.answer
    assert result.trace["generator"] == "template_combined"
