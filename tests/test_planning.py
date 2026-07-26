from __future__ import annotations

from datetime import date

from src.core_ai.nextrip_agent.planning import (
    ItineraryPlan,
    ItineraryPlanDraft,
    PlannedDay,
    PlannedStop,
    is_itinerary_request,
    planning_agent_node,
)
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.weather import (
    WeatherAssessment,
    supported_city_from_coordinates,
)


CITY = "Quy Nhơn"


def _candidate(place_id: str, entity_type: str, *, city: str = CITY, indoor=None):
    attributes = {"opening_hours_open": "06:00", "opening_hours_close": "23:00"}
    if indoor is not None:
        attributes["indoor"] = indoor
    return {
        "place_id": place_id,
        "name": place_id,
        "city": city,
        "entity_type": entity_type,
        "category": entity_type,
        "attributes": attributes,
    }


def _candidates():
    return [
        _candidate("attr_1", "attraction", indoor=True),
        _candidate("attr_2", "attraction", indoor=True),
        _candidate("attr_3", "attraction", indoor=True),
        _candidate("attr_4", "attraction", indoor=True),
        _candidate("rest_1", "restaurant"),
        _candidate("rest_2", "restaurant"),
        _candidate("rest_3", "restaurant"),
        _candidate("rest_4", "restaurant"),
        _candidate("cafe_1", "cafe"),
        _candidate("cafe_2", "cafe"),
        _candidate("hotel_1", "hotel"),
        _candidate("wrong_city", "attraction", city="Đà Nẵng"),
    ]


def test_gemini_planning_draft_avoids_unsupported_schema_keywords() -> None:
    schema_text = str(ItineraryPlanDraft.model_json_schema())

    assert "pattern" not in schema_text
    assert "minLength" not in schema_text
    assert "maxLength" not in schema_text
    assert "minItems" not in schema_text
    assert "maxItems" not in schema_text


class FakePlanner:
    def plan_itinerary(self, **kwargs):
        assert kwargs["city"] == CITY
        assert kwargs["duration_days"] == 2
        assert all(item["city"] == CITY for item in kwargs["candidates"])
        return ItineraryPlan(
            days=[
                PlannedDay(
                    day=1,
                    stops=[
                        PlannedStop(start_time="09:00", end_time="11:00", place_id="attr_1", role="activity", rationale="Tham quan."),
                        PlannedStop(start_time="11:30", end_time="13:00", place_id="rest_1", role="meal", rationale="Ăn trưa."),
                        PlannedStop(start_time="14:00", end_time="16:00", place_id="attr_2", role="activity", rationale="Vui chơi."),
                        PlannedStop(start_time="16:30", end_time="17:30", place_id="cafe_1", role="cafe_break", rationale="Nghỉ nhẹ."),
                        PlannedStop(start_time="20:00", end_time="20:30", place_id="hotel_1", role="check_in", rationale="Nhận phòng."),
                    ],
                ),
                PlannedDay(
                    day=2,
                    stops=[
                        PlannedStop(start_time="08:00", end_time="08:30", place_id="hotel_1", role="check_out", rationale="Trả phòng."),
                        PlannedStop(start_time="09:00", end_time="11:00", place_id="attr_3", role="activity", rationale="Tham quan."),
                        PlannedStop(start_time="11:30", end_time="13:00", place_id="rest_2", role="meal", rationale="Ăn trưa."),
                        PlannedStop(start_time="14:00", end_time="16:00", place_id="attr_4", role="activity", rationale="Vui chơi."),
                        PlannedStop(start_time="16:30", end_time="17:30", place_id="cafe_2", role="cafe_break", rationale="Nghỉ nhẹ."),
                    ],
                ),
            ]
        )


def test_itinerary_request_recognizes_lo_trinh() -> None:
    assert is_itinerary_request("Tôi ở Quy Nhơn 2 ngày 1 đêm, lộ trình thế nào?")


def test_planning_agent_builds_grounded_balanced_city_scoped_itinerary() -> None:
    graph = AgentResult(
        answer="",
        answer_type="recommendation",
        evidence=_candidates(),
        query_plan={"intent": "plan_candidates", "duration_days": 2},
    )

    result, trace = planning_agent_node(
        message="Tôi ở Quy Nhơn 2 ngày 1 đêm, lộ trình thế nào?",
        graph=graph,
        weather_forecast=[],
        planner=FakePlanner(),
        city=CITY,
        latitude=None,
        longitude=None,
    )

    assert result.answer_type == "itinerary_planning"
    assert len(result.itinerary) == 2
    assert all(
        slot["city"] == CITY
        for day in result.itinerary
        for slot in day["slots"]
    )
    assert {slot["role"] for slot in result.itinerary[0]["slots"]} >= {
        "activity",
        "meal",
        "cafe_break",
        "check_in",
    }
    assert trace["planner"] == "gemini_structured"


def test_planning_fallback_drops_wrong_city_and_outdoor_places_in_bad_weather() -> None:
    candidates = _candidates() + [_candidate("outdoor", "attraction", indoor=False)]
    graph = AgentResult(
        answer="",
        evidence=candidates,
        query_plan={"intent": "plan_candidates", "duration_days": 2},
    )
    weather = WeatherAssessment(
        location=CITY,
        forecast_date=date(2026, 7, 26),
        condition="Có giông",
        suitability="unsuitable",
        advice="Ưu tiên trong nhà.",
    )

    result, trace = planning_agent_node(
        message="Lên lịch trình Quy Nhơn 2 ngày 1 đêm",
        graph=graph,
        weather_forecast=[weather, weather],
        planner=None,
        city=CITY,
        latitude=None,
        longitude=None,
    )

    ids = {
        slot["place_id"]
        for day in result.itinerary
        for slot in day["slots"]
    }
    assert "wrong_city" not in ids
    assert "outdoor" not in ids
    assert trace["planner"] == "deterministic_fallback"


def test_current_coordinates_resolve_nearest_supported_city() -> None:
    assert supported_city_from_coordinates(13.78, 109.22) == CITY
