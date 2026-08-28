from __future__ import annotations

from src.apis.domains.chat.schemas import ChatResponse
from src.app import app
from src.core_ai.nextrip_agent.schemas import TypedKbPayload


def test_v8_kb_payload_maps_to_existing_be_contract() -> None:
    payload = TypedKbPayload.model_validate(
        {
            "kb_version": "v8",
            "answer_type": "recommendation",
            "recommendations": [
                {
                    "place_id": "hotel_qn_001",
                    "name": "Hotel Canonical",
                    "city": "Quy Nhơn",
                    "entity_type": "hotel",
                    "attributes": {},
                }
            ],
            "required_tools": ["live_price", "transport"],
            "query_plan": {"duration_days": 2},
        }
    )

    assert payload.recommendations[0]["place_id"] == "hotel_qn_001"
    assert not payload.recommendations[0]["place_id"].startswith("v8:")


def test_chat_response_keeps_fe_fields_and_adds_optional_current_context() -> None:
    response = ChatResponse.model_validate(
        {
            "session_id": "session-1",
            "message_id": "message-1",
            "answer": "Lịch trình",
            "kb_version": "v8",
            "itinerary": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "order": 1,
                            "start_time": "08:00",
                            "end_time": "09:00",
                            "place_id": "hotel_qn_001",
                            "name": "Hotel Canonical",
                            "city": "Quy Nhơn",
                            "entity_type": "hotel",
                            "rationale": "Check-out",
                            "hotel_availability": {
                                "selected_window_index": 0,
                            },
                            "transport_to_next": {
                                "status": "recommended",
                                "origin_place_id": "hotel_qn_001",
                                "destination_place_id": "attr_qn_001",
                                "departure_time": "2026-09-01T09:00:00+07:00",
                                "recommended_mode": "drive",
                                "distance_meters": 12000,
                                "duration_seconds": 1200,
                                "provider": "here",
                            },
                        }
                    ],
                }
            ],
        }
    ).model_dump(mode="json")

    slot = response["itinerary"][0]["slots"][0]
    assert slot["place_id"] == "hotel_qn_001"
    assert slot["transport_to_next"]["provider"] == "here"
    assert slot["hotel_availability"]["selected_window_index"] == 0
    assert response["planning"]["status"] == "skipped"


def test_openapi_exposes_the_nested_itinerary_contract_for_fe_generation() -> None:
    schemas = app.openapi()["components"]["schemas"]

    assert "ItineraryDay" in schemas
    assert "ItinerarySlot" in schemas
    assert "TransportToNext" in schemas
    assert "PlanningOutcome" in schemas
    assert "transport_to_next" in schemas["ItinerarySlot"]["properties"]
    assert "hotel_availability" in schemas["ItinerarySlot"]["properties"]
