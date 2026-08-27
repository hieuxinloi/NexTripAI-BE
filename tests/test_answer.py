from datetime import date

from src.core_ai.nextrip_agent.nodes.answer import (
    _clarification_answer,
    _display_missing_fields,
)
from src.core_ai.nextrip_agent.answer_generation import fact_display_text
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.synthesizer import synthesize_answer
from src.core_ai.nextrip_agent.weather import WeatherAssessment


def test_free_fact_is_rendered_without_currency() -> None:
    assert fact_display_text("free", "VND") == "Miễn phí"
    assert fact_display_text("25000", "VND") == "25000 VND"


def test_missing_fields_are_rendered_as_user_facing_labels() -> None:
    assert _display_missing_fields(
        ["query_constraints", "distance_between:Quy Nhơn:Đà Nẵng"]
    ) == [
        "tiêu chí hoặc điều kiện ưu tiên",
        "khoảng cách giữa Quy Nhơn và Đà Nẵng",
    ]


class ItineraryAnswerGeneratorMustNotRun:
    def synthesize(self, **kwargs):
        raise AssertionError("structured itinerary must not invoke answer Gemini")

    def generate(self, **kwargs):
        raise AssertionError("structured itinerary must not invoke answer Gemini")


def test_structured_itinerary_uses_deterministic_intro_without_second_llm_call() -> None:
    graph = AgentResult(
        answer="Upstream narrative that must not be repeated.",
        answer_type="itinerary_planning",
        evidence=[{
            "place_id": "attr_qn_001",
            "name": "Attraction Test",
            "city": "Quy Nhon",
            "entity_type": "attraction",
        }],
        itinerary=[{
            "day": 1,
            "slots": [{
                "order": 1,
                "start_time": "09:00",
                "end_time": "10:00",
                "place_id": "attr_qn_001",
                "name": "Attraction Test",
                "city": "Quy Nhon",
                "entity_type": "attraction",
            }],
        }],
    )
    weather = WeatherAssessment(
        location="Quy Nhon",
        forecast_date=date(2026, 8, 28),
        condition="Co may",
        min_temperature_c=25,
        max_temperature_c=31,
        precipitation_probability=20,
        suitability="suitable",
        advice="Phu hop de tham quan.",
    )

    result = synthesize_answer(
        question="Len lich trinh mot ngay",
        kb_version="v8",
        graph=graph,
        graph_used=True,
        weather=weather,
        weather_requested=True,
        weather_trace={"node": "weather", "status": "completed"},
        answer_generator=ItineraryAnswerGeneratorMustNotRun(),
    )

    assert result.trace["generator"] == "deterministic_itinerary_intro"
    assert "Lịch trình 1 ngày" in result.answer
    assert "28/08" in result.answer
    assert "Attraction Test" not in result.answer
    assert "09:00" not in result.answer


def test_missing_city_uses_destination_choice_question() -> None:
    assert _clarification_answer(["city"]) == (
        "Bạn muốn đi Quy Nhơn, Đà Nẵng hay xem gợi ý ở cả hai thành phố?"
    )
