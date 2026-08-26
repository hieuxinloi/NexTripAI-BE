from __future__ import annotations

from datetime import date

from src.core_ai.nextrip_agent.planning import (
    ItineraryPlan,
    ItineraryPlanDraft,
    PlanningPolicy,
    PlannedDay,
    PlannedStop,
    PreferredPeriod,
    SemanticAssignment,
    SemanticDay,
    SemanticItineraryPlan,
    SemanticItineraryPlanDraft,
    SemanticStay,
    is_itinerary_request,
    planning_agent_node,
)
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.weather import (
    WeatherAssessment,
    supported_city_from_coordinates,
)


CITY = "Quy Nhơn"


def _candidate(
    place_id: str,
    entity_type: str,
    *,
    city: str = CITY,
    indoor=None,
    lat: float | None = None,
    lng: float | None = None,
    opens: str = "06:00",
    closes: str = "23:00",
):
    attributes = {
        "opening_hours_open": opens,
        "opening_hours_close": closes,
    }
    if indoor is not None:
        attributes["indoor"] = indoor
    if lat is not None:
        attributes["lat"] = lat
    if lng is not None:
        attributes["lng"] = lng
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
    schema_text = str(ItineraryPlanDraft.model_json_schema()) + str(
        SemanticItineraryPlanDraft.model_json_schema()
    )

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
                        PlannedStop(
                            start_time="09:00",
                            end_time="11:00",
                            place_id="attr_1",
                            role="activity",
                            rationale="Tham quan.",
                        ),
                        PlannedStop(
                            start_time="11:30",
                            end_time="13:00",
                            place_id="rest_1",
                            role="meal",
                            rationale="Ăn trưa.",
                        ),
                        PlannedStop(
                            start_time="14:00",
                            end_time="16:00",
                            place_id="attr_2",
                            role="activity",
                            rationale="Vui chơi.",
                        ),
                        PlannedStop(
                            start_time="16:30",
                            end_time="17:30",
                            place_id="cafe_1",
                            role="cafe_break",
                            rationale="Nghỉ nhẹ.",
                        ),
                        PlannedStop(
                            start_time="20:00",
                            end_time="20:30",
                            place_id="hotel_1",
                            role="check_in",
                            rationale="Nhận phòng.",
                        ),
                    ],
                ),
                PlannedDay(
                    day=2,
                    stops=[
                        PlannedStop(
                            start_time="08:00",
                            end_time="08:30",
                            place_id="hotel_1",
                            role="check_out",
                            rationale="Trả phòng.",
                        ),
                        PlannedStop(
                            start_time="09:00",
                            end_time="11:00",
                            place_id="attr_3",
                            role="activity",
                            rationale="Tham quan.",
                        ),
                        PlannedStop(
                            start_time="11:30",
                            end_time="13:00",
                            place_id="rest_2",
                            role="meal",
                            rationale="Ăn trưa.",
                        ),
                        PlannedStop(
                            start_time="14:00",
                            end_time="16:00",
                            place_id="attr_4",
                            role="activity",
                            rationale="Vui chơi.",
                        ),
                        PlannedStop(
                            start_time="16:30",
                            end_time="17:30",
                            place_id="cafe_2",
                            role="cafe_break",
                            rationale="Nghỉ nhẹ.",
                        ),
                    ],
                ),
            ]
        )


class FakeSemanticPlanner:
    def plan_semantic_itinerary(self, **kwargs):
        assert kwargs["city"] == CITY
        assert kwargs["duration_days"] == 3
        return SemanticItineraryPlan(
            days=[
                SemanticDay(
                    day=1,
                    assignments=[
                        SemanticAssignment(
                            place_id="attr_1",
                            role="activity",
                            preferred_period=PreferredPeriod.MORNING,
                            rationale="Khởi đầu bằng văn hóa.",
                        ),
                        SemanticAssignment(
                            place_id="rest_1",
                            role="meal",
                            preferred_period=PreferredPeriod.LUNCH,
                            rationale="Ăn trưa gần nhóm điểm ngày đầu.",
                        ),
                        SemanticAssignment(
                            place_id="cafe_1",
                            role="cafe_break",
                            preferred_period=PreferredPeriod.AFTERNOON,
                            rationale="Nghỉ nhẹ buổi chiều.",
                        ),
                    ],
                ),
                SemanticDay(
                    day=2,
                    assignments=[
                        SemanticAssignment(
                            place_id="attr_2",
                            role="activity",
                            preferred_period=PreferredPeriod.MORNING,
                            rationale="Tiếp tục trải nghiệm văn hóa.",
                        ),
                        SemanticAssignment(
                            place_id="rest_2",
                            role="meal",
                            preferred_period=PreferredPeriod.LUNCH,
                            rationale="Bữa trưa ngày hai.",
                        ),
                        SemanticAssignment(
                            place_id="cafe_2",
                            role="cafe_break",
                            preferred_period=PreferredPeriod.AFTERNOON,
                            rationale="Nghỉ nhẹ ngày hai.",
                        ),
                    ],
                ),
                SemanticDay(
                    day=3,
                    assignments=[
                        SemanticAssignment(
                            place_id="attr_3",
                            role="activity",
                            preferred_period=PreferredPeriod.MORNING,
                            rationale="Hoạt động cuối chuyến đi.",
                        ),
                        SemanticAssignment(
                            place_id="rest_3",
                            role="meal",
                            preferred_period=PreferredPeriod.LUNCH,
                            rationale="Bữa trưa cuối chuyến đi.",
                        ),
                    ],
                ),
            ],
            stay=SemanticStay(hotel_id="hotel_1"),
            summary="Ba ngày cân bằng.",
        )


class FakePlanningRoutes:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def recommend_transport(self, **kwargs):
        self.calls.append(kwargs)
        duration = (
            100 * 60
            if kwargs["origin_id"] == "attr_1" and kwargs["destination_id"] == "rest_1"
            else 10 * 60
        )
        return {
            "status": "recommended",
            "recommended_mode": "drive",
            "selection_reason": "fastest_route_duration",
            "degraded": False,
            "partial": False,
            "options": [
                {
                    "mode": "drive",
                    "status": "eligible",
                    "recommended": True,
                    "distance_meters": 12000,
                    "duration_seconds": duration,
                    "route": {
                        "route": {
                            "provider": "here",
                            "traffic_basis": "current",
                            "traffic_aware": True,
                        }
                    },
                }
            ],
        }


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
        slot["city"] == CITY for day in result.itinerary for slot in day["slots"]
    )
    assert {slot["role"] for slot in result.itinerary[0]["slots"]} >= {
        "activity",
        "meal",
        "cafe_break",
        "check_in",
    }
    assert trace["planner"] == "gemini_structured"


def test_hybrid_planner_schedules_semantics_and_keeps_one_hotel_stay() -> None:
    graph = AgentResult(
        answer="",
        answer_type="recommendation",
        evidence=_candidates(),
        query_plan={"intent": "plan_candidates", "duration_days": 3},
    )

    result, trace = planning_agent_node(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        graph=graph,
        weather_forecast=[],
        planner=FakeSemanticPlanner(),
        city=CITY,
        latitude=None,
        longitude=None,
    )

    hotel_slots = [
        (day["day"], slot["role"], slot["place_id"])
        for day in result.itinerary
        for slot in day["slots"]
        if slot["entity_type"] == "hotel"
    ]
    assert trace["planner"] == "gemini_hybrid"
    assert hotel_slots == [
        (1, "check_in", "hotel_1"),
        (3, "check_out", "hotel_1"),
    ]
    first_day_roles = [slot["role"] for slot in result.itinerary[0]["slots"]]
    assert first_day_roles == ["activity", "meal", "check_in", "cafe_break"]
    check_in = next(
        slot for slot in result.itinerary[0]["slots"] if slot["role"] == "check_in"
    )
    check_out = next(
        slot for slot in result.itinerary[2]["slots"] if slot["role"] == "check_out"
    )
    assert check_in["start_time"] == "14:00"
    assert check_out["end_time"] <= "12:00"
    assert all(
        any(slot["role"] == "activity" for slot in day["slots"])
        for day in result.itinerary
    )


def test_hybrid_scheduler_prefers_hotel_policy_over_default_times() -> None:
    candidates = _candidates()
    hotel = next(item for item in candidates if item["entity_type"] == "hotel")
    hotel["attributes"].update(
        {
            "check_in_time": "15:00",
            "check_out_time": "09:00",
        }
    )
    graph = AgentResult(
        answer="",
        evidence=candidates,
        query_plan={"intent": "plan_candidates", "duration_days": 3},
    )

    result, trace = planning_agent_node(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        graph=graph,
        weather_forecast=[],
        planner=FakeSemanticPlanner(),
        city=CITY,
        latitude=None,
        longitude=None,
    )

    check_in = next(
        slot for slot in result.itinerary[0]["slots"] if slot["role"] == "check_in"
    )
    check_out = next(
        slot for slot in result.itinerary[2]["slots"] if slot["role"] == "check_out"
    )
    assert trace["planner"] == "gemini_hybrid"
    assert check_in["start_time"] == "15:00"
    assert check_out["end_time"] <= "09:00"


def test_hybrid_scheduler_uses_injected_policy_instead_of_fixed_clock_template() -> (
    None
):
    graph = AgentResult(
        answer="",
        evidence=_candidates(),
        query_plan={"intent": "plan_candidates", "duration_days": 3},
    )
    policy = PlanningPolicy(
        day_start_minutes=9 * 60,
        morning_end_minutes=12 * 60,
        activity_duration_minutes=60,
    )

    result, _ = planning_agent_node(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        graph=graph,
        weather_forecast=[],
        planner=FakeSemanticPlanner(),
        city=CITY,
        latitude=None,
        longitude=None,
        policy=policy,
    )

    first = result.itinerary[0]["slots"][0]
    assert first["start_time"] == "09:00"
    assert first["end_time"] == "10:00"


def test_hybrid_scheduler_uses_actual_route_duration_before_fixing_next_start() -> None:
    graph = AgentResult(
        answer="",
        evidence=_candidates(),
        query_plan={"intent": "plan_candidates", "duration_days": 3},
    )
    routes = FakePlanningRoutes()

    result, trace = planning_agent_node(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        graph=graph,
        weather_forecast=[],
        planner=FakeSemanticPlanner(),
        city=CITY,
        latitude=None,
        longitude=None,
        route_provider=routes,
        travel_date=date(2026, 8, 26),
    )

    first_day = result.itinerary[0]["slots"]
    assert trace["planner"] == "gemini_hybrid"
    assert first_day[0]["transport_to_next"]["duration_seconds"] == 6000
    assert first_day[1]["start_time"] == "11:25"
    assert first_day[0]["transport_to_next"]["provider"] == "here"
    assert routes.calls


def test_hybrid_rejects_unavailable_semantic_hotel_and_falls_back_to_available() -> (
    None
):
    candidates = _candidates()
    candidates[10]["attributes"]["hotel_availability"] = {
        "selected_window_index": None,
        "windows": [{"availability": "unavailable", "offers": []}],
    }
    available_hotel = _candidate("hotel_2", "hotel")
    available_hotel["attributes"]["hotel_availability"] = {
        "selected_window_index": 0,
        "windows": [{"availability": "available", "offers": [{"amount": 900000}]}],
    }
    candidates.append(available_hotel)
    graph = AgentResult(
        answer="",
        evidence=candidates,
        query_plan={"intent": "plan_candidates", "duration_days": 3},
    )

    result, trace = planning_agent_node(
        message="Lên lịch trình Quy Nhơn 3 ngày 2 đêm",
        graph=graph,
        weather_forecast=[],
        planner=FakeSemanticPlanner(),
        city=CITY,
        latitude=None,
        longitude=None,
    )

    hotel_ids = {
        slot["place_id"]
        for day in result.itinerary
        for slot in day["slots"]
        if slot["entity_type"] == "hotel"
    }
    assert trace["planner"] == "deterministic_fallback"
    assert hotel_ids == {"hotel_2"}
    assert any(
        warning.startswith("planning_fallback:ValueError")
        for warning in result.warnings
    )


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

    ids = {slot["place_id"] for day in result.itinerary for slot in day["slots"]}
    assert "wrong_city" not in ids
    assert "outdoor" not in ids
    assert trace["planner"] == "deterministic_fallback"


def test_planning_fallback_does_not_repeat_a_place_to_fill_empty_days() -> None:
    graph = AgentResult(
        answer="",
        evidence=[_candidate("attr_only", "attraction", indoor=True)],
        query_plan={"intent": "plan_candidates", "duration_days": 2},
    )

    result, trace = planning_agent_node(
        message="LÃªn lá»‹ch trÃ¬nh Quy NhÆ¡n 2 ngÃ y",
        graph=graph,
        weather_forecast=[],
        planner=None,
        city=CITY,
        latitude=None,
        longitude=None,
    )

    assert trace["status"] == "unavailable"
    assert trace["reason"] == "fallback_plan_invalid"
    assert result.itinerary == []


def test_planning_fallback_skips_a_restaurant_closed_during_lunch() -> None:
    candidates = [
        _candidate("attr_1", "attraction", indoor=True),
        _candidate("rest_1_closed", "restaurant", opens="16:00"),
        _candidate("rest_2_lunch", "restaurant", opens="10:00"),
        _candidate("cafe_1", "cafe"),
    ]
    graph = AgentResult(
        answer="",
        evidence=candidates,
        query_plan={"intent": "plan_candidates", "duration_days": 1},
    )

    result, trace = planning_agent_node(
        message="Lên lịch trình Quy Nhơn 1 ngày",
        graph=graph,
        weather_forecast=[],
        planner=None,
        city=CITY,
        latitude=None,
        longitude=None,
        travel_date=date(2026, 8, 26),
    )

    meal_slots = [
        slot
        for day in result.itinerary
        for slot in day["slots"]
        if slot["role"] == "meal"
    ]
    assert trace["status"] == "completed"
    assert meal_slots[0]["place_id"] == "rest_2_lunch"
    assert meal_slots[0]["start_time"] < "14:00"
    assert meal_slots[1]["place_id"] == "rest_1_closed"
    assert meal_slots[1]["start_time"] >= "17:30"


def test_planning_excludes_an_unrequested_remote_place_from_city_plan() -> None:
    candidates = [
        _candidate(
            "attr_remote",
            "attraction",
            indoor=True,
            lat=13.9205616,
            lng=108.9208407,
        ),
        _candidate("attr_near", "attraction", indoor=True, lat=13.78, lng=109.22),
        _candidate("rest_1", "restaurant", lat=13.779, lng=109.225),
        _candidate("cafe_1", "cafe", lat=13.77, lng=109.23),
        _candidate("hotel_1", "hotel", lat=13.775, lng=109.224),
    ]
    graph = AgentResult(
        answer="",
        evidence=candidates,
        query_plan={"intent": "plan_candidates", "duration_days": 1},
    )

    result, trace = planning_agent_node(
        message="Lên lịch trình một ngày ở Quy Nhơn",
        graph=graph,
        weather_forecast=[],
        planner=None,
        city=CITY,
        latitude=None,
        longitude=None,
    )

    ids = {slot["place_id"] for day in result.itinerary for slot in day["slots"]}
    assert trace["status"] == "completed"
    assert "attr_remote" not in ids
    assert "attr_near" in ids
    assert "remote_candidates_excluded:1" in result.warnings


def test_planning_keeps_a_remote_place_when_user_explicitly_requests_it() -> None:
    remote = _candidate(
        "attr_remote",
        "attraction",
        indoor=True,
        lat=13.9205616,
        lng=108.9208407,
    )
    remote["name"] = "Bảo tàng Quang Trung"
    candidates = [
        remote,
        _candidate("rest_1", "restaurant", lat=13.779, lng=109.225),
        _candidate("cafe_1", "cafe", lat=13.77, lng=109.23),
        _candidate("hotel_1", "hotel", lat=13.775, lng=109.224),
    ]
    graph = AgentResult(
        answer="",
        evidence=candidates,
        query_plan={"intent": "plan_candidates", "duration_days": 1},
    )

    result, trace = planning_agent_node(
        message="Lên lịch trình có Bảo tàng Quang Trung ở Quy Nhơn",
        graph=graph,
        weather_forecast=[],
        planner=None,
        city=CITY,
        latitude=None,
        longitude=None,
    )

    ids = {slot["place_id"] for day in result.itinerary for slot in day["slots"]}
    assert trace["status"] == "completed"
    assert "attr_remote" in ids
    assert not any(
        warning.startswith("remote_candidates_excluded") for warning in result.warnings
    )


def test_planning_returns_unavailable_when_candidates_cannot_be_planned() -> None:
    graph = AgentResult(
        answer="",
        evidence=[_candidate("dish_only", "dish")],
        query_plan={"intent": "plan_candidates", "duration_days": 1},
    )

    result, trace = planning_agent_node(
        message="LÃªn lá»‹ch trÃ¬nh Quy NhÆ¡n",
        graph=graph,
        weather_forecast=[],
        planner=None,
        city=CITY,
        latitude=None,
        longitude=None,
    )

    assert trace["status"] == "unavailable"
    assert trace["reason"] == "no_plannable_candidates"
    assert result.itinerary == []


def test_planning_returns_unavailable_when_weather_removes_only_activity() -> None:
    graph = AgentResult(
        answer="",
        evidence=[_candidate("outdoor_only", "attraction", indoor=False)],
        query_plan={"intent": "plan_candidates", "duration_days": 1},
    )
    weather = WeatherAssessment(
        location=CITY,
        forecast_date=date(2026, 7, 26),
        condition="MÆ°a rÃ o",
        suitability="unsuitable",
        advice="Æ¯u tiÃªn trong nhÃ .",
    )

    result, trace = planning_agent_node(
        message="LÃªn lá»‹ch trÃ¬nh Quy NhÆ¡n",
        graph=graph,
        weather_forecast=[weather],
        planner=None,
        city=CITY,
        latitude=None,
        longitude=None,
    )

    assert trace["status"] == "unavailable"
    assert trace["reason"] == "no_plannable_candidates"
    assert result.itinerary == []


def test_current_coordinates_resolve_nearest_supported_city() -> None:
    assert supported_city_from_coordinates(13.78, 109.22) == CITY
