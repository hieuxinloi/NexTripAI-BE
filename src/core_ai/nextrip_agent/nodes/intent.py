from __future__ import annotations

from loguru import logger

from src.core_ai.nextrip_agent.retrieval_plan import build_retrieval_plan
from src.core_ai.nextrip_agent.state import NexTripAgentState


def intent_node(state: NexTripAgentState) -> NexTripAgentState:
    trace = list(state.get("trace") or [])
    retrieval_plan = build_retrieval_plan(
        message=state["message"],
        entity_types=state.get("entity_types"),
        top_k=state["top_k"],
    )
    plan_payload = [
        {"query": item.query, "entity_types": item.entity_types, "top_k": item.top_k}
        for item in retrieval_plan
    ]
    trace.append(
        {
            "node": "intent",
            "status": "completed",
            "intent": "kb_retrieval",
            "retrieval_plan": plan_payload,
        }
    )
    logger.info(
        "NexTrip node intent result session_id={} request_count={} plan={}",
        state.get("session_id") or "-",
        len(plan_payload),
        plan_payload,
    )
    return {**state, "retrieval_plan": retrieval_plan, "trace": trace}
