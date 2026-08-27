from __future__ import annotations

from datetime import date, datetime

from src.core_ai.nextrip_agent.current_data import enrich_current_data
from src.core_ai.nextrip_agent.schemas import AgentResult


class FakeCurrentData:
    def __init__(self) -> None:
        self.hotel_request: dict | None = None

    def places(self, place_ids):
        assert place_ids == ["hotel_qn_001", "attr_qn_001"]
        return {
            "items": [
                {
                    "place_id": "hotel_qn_001",
                    "status": "available",
                    "current": {
                        "master_name": "Old Hotel",
                        "stale": False,
                        "place": {
                            "place_id": "hotel_qn_001",
                            "name": "Trivago Hotel Name",
                            "city": "Quy Nhơn",
                            "entity_type": "hotel",
                            "category": "hotel",
                            "business_status": "operational",
                            "price_level": 2,
                            "location": {
                                "latitude": 13.77,
                                "longitude": 109.23,
                            },
                            "opening": {
                                "local_date": "2026-08-31",
                                "status": "open_today",
                                "is_24_hours": False,
                                "opening_intervals": [
                                    {
                                        "opens_at": "16:00:00",
                                        "closes_at": "23:00:00",
                                    }
                                ],
                            },
                            "updated_at": "2026-08-25T00:00:00Z",
                        },
                    },
                }
            ]
        }

    def hotel_availability(self, **kwargs):
        self.hotel_request = kwargs
        return {
            "results": [
                {
                    "hotel_id": "hotel_qn_001",
                    "selected_window_index": 1,
                    "windows": [
                        {
                            "fallback_offset_days": 0,
                            "availability": "unavailable",
                            "offers": [],
                        },
                        {
                            "fallback_offset_days": 1,
                            "availability": "available",
                            "offers": [{"currency": "VND", "amount": "850000"}],
                        },
                    ],
                }
            ]
        }

    def recommend_transport(self, **kwargs):
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
                    "rank": 1,
                    "distance_meters": 12500,
                    "duration_seconds": 1200,
                    "reason_codes": ["route_available"],
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


def _graph() -> AgentResult:
    return AgentResult(
        answer="",
        answer_type="itinerary_planning",
        evidence=[
            {
                "place_id": "hotel_qn_001",
                "name": "Old Hotel",
                "city": "Quy Nhơn",
                "entity_type": "hotel",
                "attributes": {},
            },
            {
                "place_id": "attr_qn_001",
                "name": "Eo Gió",
                "city": "Quy Nhơn",
                "entity_type": "attraction",
                "attributes": {},
            },
        ],
        query_plan={"duration_days": 3},
        required_tools=["live_status", "live_price", "booking", "route", "transport"],
        itinerary=[
            {
                "day": 1,
                "slots": [
                    {
                        "order": 1,
                        "start_time": "08:00",
                        "end_time": "08:30",
                        "place_id": "hotel_qn_001",
                        "name": "Old Hotel",
                        "city": "Quy Nhơn",
                        "entity_type": "hotel",
                        "rationale": "Check-out",
                    },
                    {
                        "order": 2,
                        "start_time": "09:00",
                        "end_time": "11:00",
                        "place_id": "attr_qn_001",
                        "name": "Eo Gió",
                        "city": "Quy Nhơn",
                        "entity_type": "attraction",
                        "rationale": "Visit",
                    },
                ],
            }
        ],
    )


def test_enrichment_preserves_contract_and_attaches_current_context() -> None:
    client = FakeCurrentData()
    result, trace = enrich_current_data(
        _graph(),
        client,
        travel_date=date(2026, 8, 31),
    )

    hotel = result.evidence[0]
    assert hotel["place_id"] == "hotel_qn_001"
    assert hotel["name"] == "Trivago Hotel Name"
    assert hotel["attributes"]["current"]["business_status"] == "operational"
    assert hotel["attributes"]["lat"] == 13.77
    assert hotel["attributes"]["lng"] == 109.23
    assert hotel["attributes"]["opening_hours_open"] == "16:00:00"
    assert hotel["attributes"]["opening_hours_close"] == "23:00:00"
    assert hotel["attributes"]["hotel_availability"]["selected_window_index"] == 1
    assert client.hotel_request is not None
    assert client.hotel_request["stay_nights"] == 2
    first_slot = result.itinerary[0]["slots"][0]
    assert first_slot["hotel_availability"]["hotel_id"] == "hotel_qn_001"
    assert first_slot["transport_to_next"]["recommended_mode"] == "drive"
    assert first_slot["transport_to_next"]["provider"] == "here"
    assert result.required_tools == []
    assert trace["status"] == "completed"


class NoRepeatedLookupCurrentData:
    def places(self, place_ids):
        assert place_ids == ["attr_qn_001"]
        return {"items": []}

    def hotel_availability(self, **kwargs):
        raise AssertionError("hotel availability lookup must be reused")

    def recommend_transport(self, **kwargs):
        raise AssertionError("traffic must be disabled during pre-planning enrichment")


def test_preplanning_enrichment_reuses_current_and_hotel_context() -> None:
    enriched, _ = enrich_current_data(
        _graph(),
        FakeCurrentData(),
        travel_date=date(2026, 8, 31),
        include_traffic=False,
    )

    result, trace = enrich_current_data(
        enriched,
        NoRepeatedLookupCurrentData(),
        travel_date=date(2026, 8, 31),
        include_traffic=False,
    )

    assert (
        result.evidence[0]["attributes"]["hotel_availability"]["selected_window_index"]
        == 1
    )
    assert trace["completed"] == ["places", "hotel_availability"]
    assert trace["traffic_enabled"] is False


class RefreshedOccupancyCurrentData:
    def __init__(self) -> None:
        self.hotel_request: dict | None = None

    def places(self, place_ids):
        return {"items": []}

    def hotel_availability(self, **kwargs):
        self.hotel_request = kwargs
        return {
            "results": [
                {
                    "hotel_id": "hotel_qn_001",
                    "selected_window_index": 0,
                    "windows": [
                        {
                            "availability": "available",
                            "offers": [
                                {
                                    "currency": "VND",
                                    "total_amount": 2_400_000,
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    def recommend_transport(self, **kwargs):
        raise AssertionError("traffic must remain disabled")


def test_force_hotel_refresh_discards_stale_occupancy_context() -> None:
    enriched, _ = enrich_current_data(
        _graph(),
        FakeCurrentData(),
        travel_date=date(2026, 8, 31),
        include_traffic=False,
    )
    client = RefreshedOccupancyCurrentData()

    result, _ = enrich_current_data(
        enriched,
        client,
        travel_date=date(2026, 8, 31),
        include_traffic=False,
        adults=4,
        children=1,
        rooms=2,
        force_hotel_refresh=True,
    )

    assert client.hotel_request is not None
    assert client.hotel_request["adults"] == 4
    assert client.hotel_request["children"] == 1
    assert client.hotel_request["rooms"] == 2
    availability = result.evidence[0]["attributes"]["hotel_availability"]
    assert availability["windows"][0]["offers"][0]["total_amount"] == 2_400_000


def test_postplanning_enrichment_reuses_a_route_already_computed_by_scheduler() -> None:
    enriched, _ = enrich_current_data(
        _graph(),
        FakeCurrentData(),
        travel_date=date(2026, 8, 31),
    )

    result, trace = enrich_current_data(
        enriched,
        NoRepeatedLookupCurrentData(),
        travel_date=date(2026, 8, 31),
    )

    route = result.itinerary[0]["slots"][0]["transport_to_next"]
    assert route["provider"] == "here"
    assert trace["route_count"] == 1
    assert trace["failures"] == []


class FailingCurrentData:
    def places(self, place_ids):
        raise ConnectionError("offline")

    def hotel_availability(self, **kwargs):
        raise ConnectionError("offline")

    def recommend_transport(self, **kwargs):
        raise ConnectionError("offline")


def test_enrichment_is_fail_soft_and_keeps_original_data() -> None:
    original = _graph()
    result, trace = enrich_current_data(
        original,
        FailingCurrentData(),
        travel_date=date(2026, 8, 31),
    )

    assert result.evidence[0]["name"] == "Old Hotel"
    assert (
        result.itinerary[0]["slots"][0]["transport_to_next"]["status"] == "unavailable"
    )
    assert result.required_tools == original.required_tools
    assert result.warnings == ["current_data_partial"]
    assert trace["status"] == "unavailable"


class DirectRouteCurrentData:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.route_request: dict | None = None

    def places(self, place_ids):
        return {"items": []}

    def hotel_availability(self, **kwargs):
        raise AssertionError("hotel availability must not run for a route question")

    def recommend_transport(self, **kwargs):
        self.route_request = kwargs
        if self.fail:
            raise ConnectionError("traffic unavailable")
        return {
            "status": "recommended",
            "recommended_mode": "two_wheeler",
            "selection_reason": "balanced",
            "degraded": False,
            "partial": False,
            "options": [
                {
                    "mode": "two_wheeler",
                    "status": "eligible",
                    "recommended": True,
                    "rank": 1,
                    "distance_meters": 129400,
                    "duration_seconds": 8940,
                    "reason_codes": ["route_available"],
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


def _direct_route_graph() -> AgentResult:
    return AgentResult(
        answer="",
        answer_type="unsupported",
        evidence=[
            {
                "place_id": "attr_qn_041",
                "name": "Bảo tàng Quang Trung",
                "entity_type": "place",
                "attributes": {},
            },
            {
                "place_id": "rest_qn_021",
                "name": "GoGi House An Dương Vương Quy Nhơn",
                "entity_type": "place",
                "attributes": {},
            },
        ],
        required_tools=["route", "transport"],
    )


def test_explicit_route_question_calls_traffic_on_demand() -> None:
    client = DirectRouteCurrentData()

    result, trace = enrich_current_data(
        _direct_route_graph(),
        client,
        travel_date=None,
    )

    assert client.route_request is not None
    assert client.route_request["origin_id"] == "attr_qn_041"
    assert client.route_request["destination_id"] == "rest_qn_021"
    assert isinstance(client.route_request["departure_time"], datetime)
    route = result.evidence[0]["attributes"]["transport_to_destination"]
    assert route["recommended_mode"] == "two_wheeler"
    assert route["distance_meters"] == 129400
    assert route["duration_seconds"] == 8940
    assert route["provider"] == "here"
    assert result.facts[0]["predicate"] == "route_recommendation"
    assert "129.4 km" in result.facts[0]["value"]
    assert "149 phút" in result.facts[0]["value"]
    assert result.required_tools == []
    assert trace["completed"] == ["places", "traffic"]
    assert trace["route_count"] == 1


def test_explicit_route_failure_keeps_tools_unresolved() -> None:
    client = DirectRouteCurrentData(fail=True)

    result, trace = enrich_current_data(
        _direct_route_graph(),
        client,
        travel_date=None,
    )

    assert result.required_tools == ["route", "transport"]
    assert result.facts == []
    assert "transport_to_destination" not in result.evidence[0]["attributes"]
    assert result.warnings == ["current_data_partial"]
    assert trace["route_count"] == 0
    assert trace["failures"] == ["traffic:ConnectionError"]
