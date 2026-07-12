from __future__ import annotations

from src.apis.domains.chat.schemas import ChatRequest
from src.apis.domains.chat.service import resolve_top_k
from src.core_ai.nextrip_agent.answer_generation import facts_for_answer
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.core_ai.nextrip_agent.nodes.answer import answer_node


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
