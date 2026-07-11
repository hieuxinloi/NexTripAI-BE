from src.core_ai.nextrip_agent.graph import run_nextrip_agent


class FakeV4KbClient:
    def query_typed(self, *, query, top_k, kb_version="v4"):
        return {
            "kb_version": "v4",
            "answer_type": "recommendation",
            "recommendations": [],
            "facts": [],
            "evidence": [],
            "missing_fields": [],
            "query_plan": {"retrieval_mode": "dynamic_search"},
            "matched_paths": [],
            "constraint_results": [],
            "required_tools": ["weather"],
            "trace": [{"step": "retrieval", "status": "ok"}],
        }


class FakeV4CitationsClient:
    def query_typed(self, *, query, top_k, kb_version="v4"):
        return {
            "kb_version": "v4",
            "answer_type": "recommendation",
            "recommendations": [
                {"place_id": "rest_qn_001", "name": "Restaurant One", "city": "Quy Nhon", "entity_type": "restaurant"},
                {"place_id": "rest_qn_002", "name": "Restaurant Two", "city": "Quy Nhon", "entity_type": "restaurant"},
            ],
            "evidence": [
                {"subject_id": "rest_qn_002", "source_name": "source-two", "url": "https://two.example"},
                {"subject_id": "rest_qn_001", "source_name": "source-one", "url": "https://one.example"},
            ],
        }


def test_v4_propagates_query_plan_and_required_tools() -> None:
    result = run_nextrip_agent(
        message="Thoi tiet Quy Nhon hom nay?",
        session_id="test-v4",
        city=None,
        entity_types=None,
        top_k=5,
        kb_client=FakeV4KbClient(),
        kb_version="v4",
    )

    assert result.required_tools == ["weather"]
    assert result.query_plan["retrieval_mode"] == "dynamic_search"
    assert "weather" in result.answer


def test_v4_maps_each_citation_to_its_candidate() -> None:
    result = run_nextrip_agent(
        message="Goi y nha hang o Quy Nhon",
        session_id="test-v4-citations",
        city=None,
        entity_types=None,
        top_k=5,
        kb_client=FakeV4CitationsClient(),
        kb_version="v4",
    )

    assert result.evidence[0]["source"]["name"] == "source-one"
    assert result.evidence[1]["source"]["name"] == "source-two"
