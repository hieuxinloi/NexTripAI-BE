from __future__ import annotations

from time import perf_counter

from loguru import logger

from src.core_ai.nextrip_agent.constants import DEFAULT_KB_VERSION, KbVersion
from src.core_ai.nextrip_agent.nodes.answer import answer_node
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
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
    kb_version: KbVersion = DEFAULT_KB_VERSION,
    answer_generator: SupportsAnswerGeneration | None = None,
    conversation_context: dict | None = None,
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
        "kb_version": kb_version,
        "conversation_context": conversation_context or {},
        "trace": [],
    }
    state = intent_node(state)
    state = knowledge_node(state, kb_client)
    state = answer_node(state, answer_generator)
    result = AgentResult(
        answer=state.get("answer") or "",
        evidence=list(state.get("evidence") or []),
        facts=list(state.get("facts") or []),
        missing_fields=list(state.get("missing_fields") or []),
        answer_type=state.get("answer_type") or "kb_retrieval",
        trace=list(state.get("trace") or []),
        error=state.get("error"),
        query_plan=dict(state.get("query_plan") or {}),
        matched_paths=list(state.get("matched_paths") or []),
        constraint_results=list(state.get("constraint_results") or []),
        required_tools=list(state.get("required_tools") or []),
        itinerary=list(state.get("itinerary") or []),
        warnings=list(state.get("warnings") or []),
        conversation_context=dict(state.get("conversation_context") or {}),
    )
    logger.info(
        "NexTrip agent end session_id={} evidence_count={} trace_events={} elapsed_ms={}",
        session_id,
        len(result.evidence),
        len(result.trace),
        int((perf_counter() - started_at) * 1000),
    )
    return result
