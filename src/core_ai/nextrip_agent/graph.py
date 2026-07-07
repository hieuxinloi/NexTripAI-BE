from __future__ import annotations

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
    return AgentResult(
        answer=state.get("answer") or "",
        evidence=list(state.get("evidence") or []),
        trace=list(state.get("trace") or []),
    )
