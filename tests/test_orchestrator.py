from __future__ import annotations

from datetime import date

from src.core_ai.nextrip_agent.orchestrator import (
    OrchestrationMode,
    TravelOrchestrator,
    build_orchestration_plan,
)
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.synthesizer import synthesize_answer
from src.core_ai.nextrip_agent.weather import WeatherAssessment
from src.infra.weather import DailyForecast


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


class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        raise AssertionError("Combined flow must use synthesize()")

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return "Mỹ Khê phù hợp tham quan; thời tiết có mây và ít khả năng mưa."


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
