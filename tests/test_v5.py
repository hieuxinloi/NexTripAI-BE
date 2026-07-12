from src.core_ai.nextrip_agent.graph import run_nextrip_agent


class FakeV5TargetClient:
    def query_typed(self, *, query, top_k, kb_version="v5"):
        return {
            "kb_version": "v5",
            "answer_type": "entity_list",
            "intent": "summarize",
            "targets": [{
                "target_id": "geo-area:city_quy_nhon:tuy-phuoc",
                "kind": "geo_area",
                "name": "Tuy Phước",
                "description": "Khu vực có các di tích Chăm.",
                "score": 1.0,
            }],
            "query_plan": {"intent": "summarize"},
            "trace": [{"step": "typed_retrieval", "status": "ok"}],
        }


def test_v5_generic_target_is_exposed_as_grounded_evidence() -> None:
    result = run_nextrip_agent(
        message="Tuy Phước có gì đặc biệt?",
        session_id="test-v5-target",
        city=None,
        entity_types=None,
        top_k=5,
        kb_client=FakeV5TargetClient(),
        kb_version="v5",
    )

    assert result.evidence[0]["place_id"] == "geo-area:city_quy_nhon:tuy-phuoc"
    assert result.evidence[0]["entity_type"] == "geo_area"
    assert result.query_plan["intent"] == "summarize"
