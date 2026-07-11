from __future__ import annotations

from time import perf_counter
from typing import Protocol

from loguru import logger

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
            request_started_at = perf_counter()
            query = _query_with_city(request.get("query") or state["message"], state.get("city"))
            entity_types = request.get("entity_types")
            top_k = request.get("top_k") or state["top_k"]
            logger.info(
                "NexTrip node knowledge request start session_id={} request_index={} query={!r} entity_types={} top_k={}",
                state.get("session_id") or "-",
                index,
                query,
                entity_types or [],
                top_k,
            )
            raw_payload = kb_client.search(
                query=query,
                city=state.get("city"),
                entity_types=entity_types,
                top_k=top_k,
            )
            payload = KbSearchPayload.model_validate(raw_payload)
            evidence.extend(payload.results)
            logger.info(
                "NexTrip node knowledge request end session_id={} request_index={} strategy={} result_count={} result_ids={} elapsed_ms={}",
                state.get("session_id") or "-",
                index,
                payload.strategy or "-",
                len(payload.results),
                [item.get("place_id") for item in payload.results],
                int((perf_counter() - request_started_at) * 1000),
            )
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
        logger.info(
            "NexTrip node knowledge completed session_id={} request_count={} evidence_count={}",
            state.get("session_id") or "-",
            len(retrieval_plan),
            len(evidence),
        )
        return {**state, "evidence": evidence, "trace": trace}
    except Exception as exc:
        logger.exception(
            "NexTrip node knowledge error session_id={} error_type={}",
            state.get("session_id") or "-",
            exc.__class__.__name__,
        )
        trace.append(
            {
                "node": "knowledge",
                "status": "error",
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
        )
        return {**state, "evidence": [], "trace": trace}
