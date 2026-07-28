from __future__ import annotations

from src.apis.domains.chat.schemas import AuthenticatedChatRequest, ChatRequest
from src.apis.domains.chat.service import handle_chat, resolve_top_k
from src.core_ai.personalization.models import PersonalizationUpdate
from src.core_ai.nextrip_agent.answer_generation import facts_for_answer
from src.core_ai.nextrip_agent.constants import (
    supports_structured_conversation_context,
)
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.core_ai.nextrip_agent.nodes.answer import answer_node
from src.infra.user_profile_store import InMemoryUserProfileStore


class FakeKbClient:
    def __init__(self) -> None:
        self.last_top_k: int | None = None
        self.calls: list[dict] = []

    def search(self, *, query, city, entity_types, top_k):
        self.last_top_k = top_k
        self.calls.append({"query": query, "entity_types": entity_types, "top_k": top_k})
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
    facts = facts_for_answer([
        {"subject_id": "hotel_qn_001", "predicate": "address", "value": "186 Xuân Diệu"},
        {"subject_id": "hotel_qn_001", "predicate": "location", "value": {"lat": 1, "lng": 2}},
        {"subject_id": "hotel_qn_001", "predicate": "phone", "value": "0123"},
    ])

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


def test_user_profile_reaches_v8_as_structured_personalization_context() -> None:
    client = FakeV8ItineraryClient()
    profiles = InMemoryUserProfileStore()
    profiles.update_profile(
        "alice",
        PersonalizationUpdate(
            budget_level="budget",
            travel_pace="relaxed",
            party_type="family",
            preferred_concepts=["beach", "local_food"],
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
        "preferred_concepts": ["beach", "local_food"],
        "excluded_concepts": ["nightclub"],
        "preferred_cities": [],
        "dietary_requirements": [],
        "accessibility_requirements": [],
        "transport_preferences": [],
    }


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

    assert "chưa có địa điểm với quan hệ vị trí đủ tin cậy tại Tuy Phước" in state["answer"]


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
