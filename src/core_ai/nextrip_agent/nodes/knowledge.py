from __future__ import annotations

from typing import Protocol

from src.core_ai.nextrip_agent.schemas import KbSearchPayload
from src.core_ai.nextrip_agent.state import NexTripAgentState


class SupportsKbSearch(Protocol):
    def search(
        self,
        *,
        query: str,
        city: str | None,
        entity_types: list[str] | None,
        top_k: int,
    ) -> dict:
        ...


def _query_with_city(query: str, city: str | None) -> str:
    if not city:
        return query
    if city.lower() in query.lower():
        return query
    return f"{query} o {city}"


def knowledge_node(state: NexTripAgentState, kb_client: SupportsKbSearch) -> NexTripAgentState:
    trace = list(state.get("trace") or [])
    evidence: list[dict] = []
    retrieval_plan = state.get("retrieval_plan") or [
        {"entity_types": state.get("entity_types"), "top_k": state["top_k"]}
    ]
    try:
        for index, request in enumerate(retrieval_plan, start=1):
            query = _query_with_city(request.get("query") or state["message"], state.get("city"))
            entity_types = request.get("entity_types")
            top_k = request.get("top_k") or state["top_k"]
            raw_payload = kb_client.search(
                query=query,
                city=state.get("city"),
                entity_types=entity_types,
                top_k=top_k,
            )
            payload = KbSearchPayload.model_validate(raw_payload)
            evidence.extend(payload.results)
            trace.append(
                {
                    "node": "knowledge",
                    "step": "kb_search",
                    "request_index": index,
                    "query": query,
                    "entity_types": entity_types,
                    "top_k": top_k,
                    "count": len(payload.results),
                }
            )
            trace.extend(payload.trace)
        trace.append({"node": "knowledge", "status": "completed", "count": len(evidence)})
        return {**state, "evidence": evidence, "trace": trace}
    except Exception as exc:
        trace.append(
            {
                "node": "knowledge",
                "status": "error",
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
        )
        return {**state, "evidence": [], "trace": trace}
