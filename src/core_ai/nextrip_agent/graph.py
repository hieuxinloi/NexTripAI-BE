from __future__ import annotations

from time import perf_counter

from loguru import logger

from src.core_ai.nextrip_agent.nodes.answer import answer_node
from src.core_ai.nextrip_agent.nodes.intent import intent_node
from src.core_ai.nextrip_agent.nodes.knowledge import SupportsKbSearch, knowledge_node
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.state import NexTripAgentState


def run_nextrip_agent(
    *,
    message: str,
    session_id: str,
    city: str | None,
    entity_types: list[str] | None,
    top_k: int,
    kb_client: SupportsKbSearch,
) -> AgentResult:
    started_at = perf_counter()
    logger.info(
        "NexTrip agent start session_id={} city={} entity_types={} top_k={}",
        session_id,
        city or "-",
        entity_types or [],
        top_k,
    )
    state: NexTripAgentState = {
        "message": message,
        "session_id": session_id,
        "city": city,
        "entity_types": entity_types,
        "top_k": top_k,
        "trace": [],
    }
    state = intent_node(state)
    state = knowledge_node(state, kb_client)
    state = answer_node(state)
    result = AgentResult(
        answer=state.get("answer") or "",
        evidence=list(state.get("evidence") or []),
        trace=list(state.get("trace") or []),
        error=state.get("error"),
    )
    logger.info(
        "NexTrip agent end session_id={} evidence_count={} trace_events={} elapsed_ms={}",
        session_id,
        len(result.evidence),
        len(result.trace),
        int((perf_counter() - started_at) * 1000),
    )
    return result
