from __future__ import annotations

from src.apis.domains.chat.service import infer_top_k
from src.core_ai.nextrip_agent.graph import run_nextrip_agent


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
    assert fake_kb.calls == [
        {"query": "cafe o Quy Nhon", "entity_types": ["cafe"], "top_k": 3},
        {"query": "nha hang o Quy Nhon", "entity_types": ["restaurant"], "top_k": 5},
    ]
    assert len(result.evidence) == 8


def test_infer_top_k_from_message() -> None:
    assert infer_top_k("Goi y 3 quan cafe o Quy Nhon") == 3


def test_infer_top_k_uses_default_without_number() -> None:
    assert infer_top_k("Goi y quan cafe o Quy Nhon") == 5
