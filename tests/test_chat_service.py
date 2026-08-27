from __future__ import annotations

import pytest

from src.apis.domains.chat.schemas import (
    AuthenticatedChatRequest,
    ChatRequest,
    EvidenceItem,
)
from src.apis.domains.chat.service import (
    _build_clarification,
    _record_grounded_place_interest,
    handle_chat,
    resolve_top_k,
)
from src.core_ai.personalization.models import PersonalizationUpdate
from src.core_ai.nextrip_agent.answer_generation import facts_for_answer
from src.core_ai.nextrip_agent.constants import (
    supports_structured_conversation_context,
)
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.core_ai.nextrip_agent.nodes.answer import answer_node
from src.infra.chat_store import InMemoryChatStore, TripPlanRevisionConflictError
from src.infra.user_profile_store import InMemoryUserProfileStore


def test_missing_city_builds_destination_choice_cards() -> None:
    clarification = _build_clarification(["city"], required_tools=[])

    assert clarification is not None
    assert clarification.field == "city"
    assert clarification.prompt == (
        "Bạn muốn đi Quy Nhơn, Đà Nẵng hay xem gợi ý ở cả hai thành phố?"
    )
    assert [(option.label, option.value) for option in clarification.options] == [
        ("Đà Nẵng", "Đà Nẵng"),
        ("Quy Nhơn", "Quy Nhơn"),
        ("Cả hai thành phố", "all"),
    ]


class FakeKbClient:
    def __init__(self) -> None:
        self.last_top_k: int | None = None
        self.calls: list[dict] = []

    def search(self, *, query, city, entity_types, top_k):
        self.last_top_k = top_k
        self.calls.append(
            {"query": query, "entity_types": entity_types, "top_k": top_k}
        )
        entity_type = entity_types[0] if entity_types else "cafe"
        return {
            "results": [
                {
                    "place_id": f"{entity_type}_qn_{index:03d}",
                    "name": f"{entity_type.title()} Test {index}",
                    "city": "Quy Nhon",
                    "entity_type": entity_type,
                    "category": "work_cafe",
                    "score": 0.9,
                    "source": {"name": "test"},
                    "graph_context": {"facets": ["wifi"], "nearby": []},
                }
                for index in range(1, top_k + 1)
            ],
            "trace": [{"step": "fake_search", "status": "ok"}],
        }


class FakeV2KbClient(FakeKbClient):
    def query_typed(self, *, query, top_k, kb_version="v2"):
        return {
            "kb_version": kb_version,
            "answer_type": "entity_detail",
            "entities": [
                {
                    "place_id": "attr_dn_016",
                    "name": "Bãi biển Mỹ Khê",
                    "city": "Đà Nẵng",
                    "entity_type": "attraction",
                    "category": "Biển đảo",
                }
            ],
            "facts": [
                {
                    "fact_id": "fact:attr_dn_016:address",
                    "subject_id": "attr_dn_016",
                    "predicate": "address",
                    "value": "Đà Nẵng 550000, Việt Nam",
                    "value_type": "string",
                    "confidence": 0.7,
                    "evidence_ids": ["text-unit:verified:attr_dn_016"],
                }
            ],
            "evidence": [],
            "missing_fields": [],
            "trace": [{"step": "lookup", "status": "ok"}],
        }


class FakeAnswerGenerator:
    def __init__(self, answer: str = "Câu trả lời được diễn đạt từ GraphRAG.") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer


class FailingAnswerGenerator:
    def generate(self, **kwargs):
        raise RuntimeError("LLM unavailable")


class FakeV8ItineraryClient(FakeKbClient):
    def __init__(self) -> None:
        super().__init__()
        self.context: dict | None = None
        self.query: str | None = None

    def query_typed(
        self,
        *,
        query,
        top_k,
        kb_version="v8",
        conversation_context=None,
    ):
        self.query = query
        self.context = conversation_context
        return {
            "kb_version": "v8",
            "answer_type": "recommendation",
            "recommendations": [
                {
                    "place_id": "attr_qn_001",
                    "name": "Eo Gió",
                    "city": "Quy Nhơn",
                    "entity_type": "attraction",
                    "category": "Biển đảo",
                }
            ],
            "itinerary": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "order": 1,
                            "start_time": "09:00",
                            "end_time": "10:30",
                            "place_id": "attr_qn_001",
                            "name": "Eo Gió",
                            "city": "Quy Nhơn",
                            "entity_type": "attraction",
                            "rationale": "Điểm tham quan đã được grounding.",
                        }
                    ],
                }
            ],
            "warnings": ["itinerary_preferences_relaxed"],
            "conversation_context": {
                "turn_count": 2,
                "cities": ["Quy Nhơn"],
            },
            "missing_fields": [],
            "trace": [],
        }


class FakeV8AddCandidateClient(FakeV8ItineraryClient):
    def query_typed(
        self,
        *,
        query,
        top_k,
        kb_version="v8",
        conversation_context=None,
    ):
        self.query = query
        self.context = conversation_context
        return {
            "kb_version": "v8",
            "answer_type": "recommendation",
            "recommendations": [
                {
                    "place_id": "cafe_qn_099",
                    "name": "Cafe Mới",
                    "city": "Quy Nhơn",
                    "entity_type": "cafe",
                    "category": "cafe",
                    "attributes": {
                        "lat": 13.77,
                        "lng": 109.22,
                        "price_per_person_min": 35_000,
                        "price_per_person_max": 60_000,
                    },
                }
            ],
            "itinerary": [],
            "missing_fields": [],
            "trace": [],
        }


class FakeV8TwoDayItineraryClient(FakeV8ItineraryClient):
    def query_typed(
        self,
        *,
        query,
        top_k,
        kb_version="v8",
        conversation_context=None,
    ):
        payload = super().query_typed(
            query=query,
            top_k=top_k,
            kb_version=kb_version,
            conversation_context=conversation_context,
        )
        payload["recommendations"].append(
            {
                "place_id": "cafe_qn_002",
                "name": "Cafe Ngày Hai",
                "city": "Quy Nhơn",
                "entity_type": "cafe",
                "category": "cafe",
            }
        )
        payload["itinerary"].append(
            {
                "day": 2,
                "slots": [
                    {
                        "order": 1,
                        "start_time": "09:30",
                        "end_time": "10:30",
                        "place_id": "cafe_qn_002",
                        "name": "Cafe Ngày Hai",
                        "city": "Quy Nhơn",
                        "entity_type": "cafe",
                        "role": "cafe_break",
                    }
                ],
            }
        )
        return payload


class FakeV8RouteClient(FakeKbClient):
    def query_typed(
        self,
        *,
        query,
        top_k,
        kb_version="v8",
        conversation_context=None,
    ):
        return {
            "kb_version": "v8",
            "answer_type": "unsupported",
            "targets": [
                {
                    "target_id": "attr_qn_041",
                    "kind": "place",
                    "name": "Bảo tàng Quang Trung",
                    "score": 1.0,
                },
                {
                    "target_id": "rest_qn_021",
                    "kind": "place",
                    "name": "GoGi House An Dương Vương Quy Nhơn",
                    "score": 1.0,
                },
            ],
            "query_plan": {
                "intent": "tool_required",
                "required_tools": ["route", "transport"],
            },
            "required_tools": ["route", "transport"],
            "missing_fields": [],
            "trace": [],
        }


class FakeRouteCurrentData:
    def __init__(self) -> None:
        self.route_request: dict | None = None

    def places(self, place_ids):
        return {"items": []}

    def hotel_availability(self, **kwargs):
        raise AssertionError("hotel lookup is unrelated to a route request")

    def recommend_transport(self, **kwargs):
        self.route_request = kwargs
        return {
            "status": "recommended",
            "recommended_mode": "two_wheeler",
            "selection_reason": "balanced_generalized_duration",
            "degraded": False,
            "partial": False,
            "options": [
                {
                    "mode": "two_wheeler",
                    "status": "eligible",
                    "recommended": True,
                    "rank": 1,
                    "distance_meters": 45600,
                    "duration_seconds": 3893,
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


def test_structured_conversation_context_is_a_version_capability() -> None:
    assert supports_structured_conversation_context("v7") is False
    assert supports_structured_conversation_context("v8") is True
    assert supports_structured_conversation_context("v9") is True
    assert supports_structured_conversation_context("latest") is False


def test_nextrip_agent_uses_kb_evidence() -> None:
    fake_kb = FakeKbClient()
    result = run_nextrip_agent(
        message="Goi y cafe o Quy Nhon",
        session_id="test-session",
        city="Quy Nhon",
        entity_types=["cafe"],
        top_k=3,
        kb_client=fake_kb,
    )
    assert "Cafe Test 1" in result.answer
    assert result.evidence[0]["place_id"] == "cafe_qn_001"
    assert fake_kb.last_top_k == 3


def test_nextrip_agent_splits_multi_entity_counts() -> None:
    fake_kb = FakeKbClient()
    result = run_nextrip_agent(
        message="Goi y 3 quan cafe va 5 nha hang o Quy Nhon",
        session_id="test-session",
        city="Quy Nhon",
        entity_types=None,
        top_k=3,
        kb_client=fake_kb,
    )
    assert sorted(fake_kb.calls, key=lambda item: item["top_k"]) == [
        {"query": "cafe o Quy Nhon", "entity_types": ["cafe"], "top_k": 3},
        {"query": "nha hang o Quy Nhon", "entity_types": ["restaurant"], "top_k": 5},
    ]
    assert len(result.evidence) == 8
    assert result.evidence[0]["entity_type"] == "cafe"
    assert result.evidence[3]["entity_type"] == "restaurant"


def test_typed_query_lets_kb_planner_choose_requested_result_count() -> None:
    request = ChatRequest(
        message="Quy Nhon 1 ngay can lam gi?",
        session_id="planner-limit",
        kb_version="v4",
    )

    assert resolve_top_k(request) == 20


def test_explicit_top_k_overrides_query_ceiling() -> None:
    request = ChatRequest(
        message="Goi y quan cafe o Quy Nhon",
        session_id="explicit-limit",
        kb_version="v4",
        top_k=7,
    )

    assert resolve_top_k(request) == 7


def test_nextrip_agent_uses_typed_v2_facts() -> None:
    result = run_nextrip_agent(
        message="Bãi biển Mỹ Khê ở đâu?",
        session_id="test-v2",
        city=None,
        entity_types=None,
        top_k=5,
        kb_client=FakeV2KbClient(),
        kb_version="v2",
    )

    assert result.answer_type == "entity_detail"
    assert "Đà Nẵng 550000" in result.answer
    assert result.facts[0]["predicate"] == "address"


def test_answer_agent_uses_grounded_generator_context() -> None:
    generator = FakeAnswerGenerator()
    result = run_nextrip_agent(
        message="Bãi biển Mỹ Khê ở đâu?",
        session_id="test-grounded-answer",
        city=None,
        entity_types=None,
        top_k=5,
        kb_client=FakeV2KbClient(),
        kb_version="v2",
        answer_generator=generator,
    )

    assert result.answer == generator.answer
    assert generator.calls[0]["facts"][0]["predicate"] == "address"
    assert result.trace[-1]["generator"] == "llm_grounded"


def test_answer_generator_omits_redundant_location_when_address_exists() -> None:
    facts = facts_for_answer(
        [
            {
                "subject_id": "hotel_qn_001",
                "predicate": "address",
                "value": "186 Xuân Diệu",
            },
            {
                "subject_id": "hotel_qn_001",
                "predicate": "location",
                "value": {"lat": 1, "lng": 2},
            },
            {"subject_id": "hotel_qn_001", "predicate": "phone", "value": "0123"},
        ]
    )

    predicates = [fact["predicate"] for fact in facts]
    assert predicates == ["address", "phone"]


def test_answer_generator_omits_coordinates_when_city_answers_address_query() -> None:
    facts = facts_for_answer(
        [
            {
                "subject_id": "v8:attr_qn_001",
                "predicate": "location",
                "value": '{"lat":13.88,"lng":109.29}',
            }
        ],
        evidence=[{"place_id": "v8:attr_qn_001", "city": "Quy Nhơn"}],
        query_plan={"requested_fields": ["address", "location"]},
    )

    assert facts == []


def test_answer_agent_falls_back_when_llm_fails() -> None:
    result = run_nextrip_agent(
        message="Bãi biển Mỹ Khê ở đâu?",
        session_id="test-answer-fallback",
        city=None,
        entity_types=None,
        top_k=5,
        kb_client=FakeV2KbClient(),
        kb_version="v2",
        answer_generator=FailingAnswerGenerator(),
    )

    assert "Đà Nẵng 550000" in result.answer
    assert result.trace[-2]["status"] == "fallback"
    assert result.trace[-1]["generator"] == "template"


def test_v8_itinerary_context_and_warnings_survive_the_agent_boundary() -> None:
    client = FakeV8ItineraryClient()
    result = run_nextrip_agent(
        message="Lên lịch trình một ngày ở Quy Nhơn",
        session_id="itinerary-v8",
        city="Quy Nhơn",
        entity_types=None,
        top_k=5,
        kb_client=client,
        kb_version="v8",
        conversation_context={
            "turn_count": 1,
            "cities": ["Quy Nhơn"],
        },
    )

    assert client.context == {
        "turn_count": 1,
        "cities": ["Quy Nhơn"],
    }
    assert result.itinerary[0]["slots"][0]["place_id"] == "attr_qn_001"
    assert result.warnings == ["itinerary_preferences_relaxed"]
    assert result.conversation_context["turn_count"] == 2


def test_chat_persists_and_revises_an_active_trip_plan() -> None:
    store = InMemoryChatStore()
    first = handle_chat(
        ChatRequest(
            message="Lên lịch trình một ngày ở Quy Nhơn",
            session_id="mutable-plan",
            city="Quy Nhơn",
            kb_version="v8",
        ),
        FakeV8ItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )

    assert first.active_trip_plan is not None
    assert first.active_trip_plan.revision == 1
    assert first.itinerary[0].slots[0].slot_id

    second = handle_chat(
        ChatRequest(
            message="Đổi ngân sách thành 2 triệu",
            session_id="mutable-plan",
            kb_version="v8",
            expected_plan_revision=1,
        ),
        FakeV8ItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )

    assert second.orchestration_mode == "plan_revision"
    assert second.active_trip_plan is not None
    assert second.active_trip_plan.revision == 2
    assert second.budget_summary is not None
    assert second.budget_summary.budget_amount == 2_000_000

    with pytest.raises(TripPlanRevisionConflictError):
        handle_chat(
            ChatRequest(
                message="Bỏ Eo Gió",
                session_id="mutable-plan",
                kb_version="v8",
                expected_plan_revision=1,
            ),
            FakeV8ItineraryClient(),
            FakeAnswerGenerator(),
            chat_store=store,
        )


def test_active_plan_is_not_overwritten_by_an_unrelated_recommendation() -> None:
    store = InMemoryChatStore()
    handle_chat(
        ChatRequest(
            message="Lên lịch trình một ngày ở Quy Nhơn",
            session_id="plan-independent-query",
            city="Quy Nhơn",
            kb_version="v8",
        ),
        FakeV8ItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )

    response = handle_chat(
        ChatRequest(
            message="Gợi ý thêm các quán trà ở Quy Nhơn",
            session_id="plan-independent-query",
            kb_version="v8",
            expected_plan_revision=1,
        ),
        FakeV8ItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )

    stored = store.get_active_trip_plan("plan-independent-query")
    assert stored is not None
    assert stored["revision"] == 1
    assert response.plan_change is None


def test_chat_add_move_and_retime_are_local_plan_revisions() -> None:
    store = InMemoryChatStore()
    first = handle_chat(
        ChatRequest(
            message="Lên lịch trình một ngày ở Quy Nhơn",
            session_id="plan-local-edits",
            city="Quy Nhơn",
            kb_version="v8",
        ),
        FakeV8ItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )
    assert first.active_trip_plan is not None

    added = handle_chat(
        ChatRequest(
            message="Thêm Cafe Mới vào ngày 1 lúc 11:00",
            session_id="plan-local-edits",
            kb_version="v8",
            expected_plan_revision=1,
        ),
        FakeV8AddCandidateClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )
    assert added.active_trip_plan is not None
    assert added.active_trip_plan.revision == 2
    assert [slot.name for slot in added.itinerary[0].slots] == ["Eo Gió", "Cafe Mới"]
    added_slot_id = added.itinerary[0].slots[1].slot_id

    moved = handle_chat(
        ChatRequest(
            message="Chuyển Cafe Mới sang ngày 1 vị trí 1",
            session_id="plan-local-edits",
            kb_version="v8",
            expected_plan_revision=2,
        ),
        FakeV8ItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )
    assert moved.active_trip_plan is not None
    assert moved.active_trip_plan.revision == 3
    assert moved.itinerary[0].slots[0].slot_id == added_slot_id

    retimed = handle_chat(
        ChatRequest(
            message="Đổi giờ Cafe Mới sang lúc 15:00",
            session_id="plan-local-edits",
            kb_version="v8",
            expected_plan_revision=3,
        ),
        FakeV8ItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )
    assert retimed.active_trip_plan is not None
    assert retimed.active_trip_plan.revision == 4
    assert retimed.itinerary[0].slots[0].start_time == "15:00"


def test_replan_day_preserves_other_days_and_their_slot_ids() -> None:
    store = InMemoryChatStore()
    first = handle_chat(
        ChatRequest(
            message="Lên lịch trình 2 ngày ở Quy Nhơn",
            session_id="replan-one-day",
            city="Quy Nhơn",
            kb_version="v8",
        ),
        FakeV8TwoDayItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )
    assert first.active_trip_plan is not None
    preserved_id = first.itinerary[0].slots[0].slot_id

    second = handle_chat(
        ChatRequest(
            message="Sắp xếp lại ngày 2",
            session_id="replan-one-day",
            kb_version="v8",
            expected_plan_revision=1,
        ),
        FakeV8TwoDayItineraryClient(),
        FakeAnswerGenerator(),
        chat_store=store,
    )

    assert second.active_trip_plan is not None
    assert second.active_trip_plan.revision == 2
    assert second.itinerary[0].slots[0].slot_id == preserved_id
    assert second.plan_change is not None
    assert second.plan_change.operation.value == "replan_day"
    assert preserved_id in second.plan_change.preserved_slot_ids


def test_chat_route_question_calls_current_traffic_with_canonical_ids() -> None:
    answer_generator = FakeAnswerGenerator("Đi xe máy, dự kiến khoảng 65 phút.")
    current_data = FakeRouteCurrentData()

    response = handle_chat(
        ChatRequest(
            message=(
                "Từ Bảo tàng Quang Trung tới GoGi House nên di chuyển bằng gì "
                "và thời gian di chuyển như nào?"
            ),
            session_id="route-on-demand",
            kb_version="v8",
        ),
        FakeV8RouteClient(),
        answer_generator,
        current_data_client=current_data,
    )

    assert current_data.route_request is not None
    assert current_data.route_request["origin_id"] == "attr_qn_041"
    assert current_data.route_request["destination_id"] == "rest_qn_021"
    assert response.required_tools == []
    assert response.facts[0]["predicate"] == "route_recommendation"
    assert (
        response.evidence[0].attributes["transport_to_destination"]["provider"]
        == "here"
    )
    assert response.answer == "Đi xe máy, dự kiến khoảng 65 phút."
    assert answer_generator.calls[0]["facts"][0]["predicate"] == "route_recommendation"


def test_user_profile_reaches_v8_as_structured_personalization_context() -> None:
    client = FakeV8ItineraryClient()
    profiles = InMemoryUserProfileStore()
    profiles.update_profile(
        "alice",
        PersonalizationUpdate(
            budget_level="budget",
            travel_pace="relaxed",
            party_type="family",
            preferred_concepts=["beach", "vietnamese"],
            excluded_concepts=["nightclub"],
        ),
    )

    handle_chat(
        AuthenticatedChatRequest(
            message="Lên lịch trình một ngày ở Quy Nhơn",
            session_id="personalized-v8",
            user_id="alice",
            kb_version="v8",
        ),
        client,
        FakeAnswerGenerator(),
        user_profile_store=profiles,
    )

    assert client.context is not None
    assert client.context["personalization"] == {
        "profile_revision": 1,
        "budget_level": "budget",
        "travel_pace": "relaxed",
        "party_type": "family",
        "hard_constraints": {},
        "preferred_concepts": ["beach", "vietnamese"],
        "excluded_concepts": ["nightclub"],
        "preferred_cities": [],
        "dietary_requirements": [],
        "accessibility_requirements": [],
        "transport_preferences": [],
    }


def test_disabled_personalization_does_not_record_implicit_place_interest() -> None:
    profiles = InMemoryUserProfileStore()

    _record_grounded_place_interest(
        profiles,
        user_id="alice",
        session_id="private-session",
        answer_type="entity_detail",
        evidence=[EvidenceItem(place_id="v8:place-1")],
        personalization_enabled=False,
    )

    assert profiles.recent_events("alice") == []


def test_v8_keeps_named_query_verbatim_instead_of_appending_context_city() -> None:
    client = FakeV8ItineraryClient()

    run_nextrip_agent(
        message="Highlight Coffee o dau",
        session_id="named-v8",
        city="Quy Nhon",
        entity_types=None,
        top_k=5,
        kb_client=client,
        kb_version="v8",
        conversation_context={
            "turn_count": 1,
            "cities": ["Quy Nhon"],
            "city_source": "conversation_history",
        },
    )

    assert client.query == "Highlight Coffee o dau"
    assert client.context == {
        "turn_count": 1,
        "cities": ["Quy Nhon"],
        "city_source": "conversation_history",
    }


def test_v2_answer_formats_count_breakdown() -> None:
    state = answer_node(
        {
            "session_id": "count-v2",
            "kb_version": "v2",
            "answer_type": "aggregate_count",
            "facts": [
                {"predicate": "count", "entity_type": "cafe", "value": 36},
                {"predicate": "count", "entity_type": "restaurant", "value": 70},
                {"predicate": "count", "entity_type": "hotel", "value": 30},
            ],
            "evidence": [],
            "trace": [],
        }
    )

    assert "Quán cafe: 36" in state["answer"]
    assert "Nhà hàng: 70" in state["answer"]
    assert "Khách sạn: 30" in state["answer"]


def test_aggregate_count_uses_verified_fact_without_llm_rewrite() -> None:
    generator = FakeAnswerGenerator("sai: 3 địa điểm ban đêm 18")

    state = answer_node(
        {
            "session_id": "count-v8",
            "kb_version": "v8",
            "answer_type": "aggregate_count",
            "query_plan": {"geo_scope": {"cities": ["Quy Nhơn"]}},
            "facts": [
                {
                    "predicate": "count",
                    "entity_type": "nightlife",
                    "value": 18,
                }
            ],
            "evidence": [],
            "trace": [],
        },
        answer_generator=generator,
    )

    assert state["answer"] == (
        "Ở Quy Nhơn có tổng cộng 18 địa điểm nightlife để bạn lựa chọn."
    )
    assert generator.calls == []


def test_v2_missing_entity_is_not_reported_as_service_failure() -> None:
    state = answer_node(
        {
            "session_id": "missing-v2",
            "kb_version": "v2",
            "answer_type": "entity_detail",
            "facts": [],
            "evidence": [],
            "trace": [],
        }
    )

    assert "Không tìm thấy địa điểm" in state["answer"]


def test_missing_v4_entity_is_reported_as_absent_data() -> None:
    state = answer_node(
        {
            "session_id": "missing-v4",
            "kb_version": "v4",
            "answer_type": "entity_detail",
            "missing_fields": ["not_found:entity:FLC"],
            "facts": [],
            "evidence": [],
            "trace": [],
        }
    )

    assert state["answer"] == "Mình chưa tìm thấy FLC trong Knowledge Base V4."


def test_missing_v5_place_is_reported_without_internal_error_code() -> None:
    state = answer_node(
        {
            "session_id": "missing-place-v5",
            "kb_version": "v5",
            "answer_type": "entity_detail",
            "missing_fields": ["not_found:place:Khách sạn Hilton Da Nang"],
            "facts": [],
            "evidence": [],
            "trace": [],
        }
    )

    assert state["answer"] == (
        "Mình chưa tìm thấy Khách sạn Hilton Da Nang trong Knowledge Base V5."
    )
    assert "not_found" not in state["answer"]


def test_unverified_geo_scope_is_reported_as_data_gap() -> None:
    state = answer_node(
        {
            "session_id": "missing-geo-v5",
            "kb_version": "v5",
            "answer_type": "entity_list",
            "missing_fields": ["verified_geo_candidates:Tuy Phước"],
            "facts": [],
            "evidence": [],
            "trace": [],
        }
    )

    assert (
        "chưa có địa điểm với quan hệ vị trí đủ tin cậy tại Tuy Phước"
        in state["answer"]
    )


def test_internal_query_constraint_is_rendered_as_natural_clarification() -> None:
    state = answer_node(
        {
            "session_id": "clarification-v5",
            "kb_version": "v5",
            "answer_type": "entity_list",
            "missing_fields": ["query_constraints"],
            "facts": [],
            "evidence": [],
            "trace": [],
        }
    )

    assert "query_constraints" not in state["answer"]
    assert "thành phố" in state["answer"]
