from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from loguru import logger

from src.core_ai.nextrip_agent.schemas import KbSearchPayload
from src.core_ai.nextrip_agent.retrieval_plan import RetrievalRequest
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


@dataclass(frozen=True)
class SearchOutcome:
    index: int
    query: str
    entity_types: list[str] | None
    top_k: int
    payload: KbSearchPayload
    elapsed_ms: int


def _query_with_city(query: str, city: str | None) -> str:
    if not city:
        return query
    if city.lower() in query.lower():
        return query
    return f"{query} o {city}"


def _run_search(
    *,
    index: int,
    request: RetrievalRequest,
    state: NexTripAgentState,
    kb_client: SupportsKbSearch,
) -> SearchOutcome:
    started_at = perf_counter()
    query = _query_with_city(request.query, state.get("city"))
    logger.info(
        "NexTrip node knowledge request start session_id={} request_index={} query={!r} entity_types={} top_k={}",
        state.get("session_id") or "-",
        index,
        query,
        request.entity_types or [],
        request.top_k,
    )
    payload = KbSearchPayload.model_validate(
        kb_client.search(
            query=query,
            city=state.get("city"),
            entity_types=request.entity_types,
            top_k=request.top_k,
        )
    )
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "NexTrip node knowledge request end session_id={} request_index={} strategy={} result_count={} result_ids={} elapsed_ms={}",
        state.get("session_id") or "-",
        index,
        payload.strategy or "-",
        len(payload.results),
        [item.get("place_id") for item in payload.results],
        elapsed_ms,
    )
    return SearchOutcome(
        index,
        query,
        request.entity_types,
        request.top_k,
        payload,
        elapsed_ms,
    )


def _execute_plan(
    retrieval_plan: list[RetrievalRequest],
    state: NexTripAgentState,
    kb_client: SupportsKbSearch,
) -> list[SearchOutcome]:
    if len(retrieval_plan) == 1:
        return [_run_search(index=1, request=retrieval_plan[0], state=state, kb_client=kb_client)]

    with ThreadPoolExecutor(max_workers=len(retrieval_plan), thread_name_prefix="kb-search") as pool:
        futures = [
            pool.submit(
                copy_context().run,
                _run_search,
                index=index,
                request=request,
                state=state,
                kb_client=kb_client,
            )
            for index, request in enumerate(retrieval_plan, start=1)
        ]
        return [future.result() for future in futures]


def knowledge_node(state: NexTripAgentState, kb_client: SupportsKbSearch) -> NexTripAgentState:
    trace = list(state.get("trace") or [])
    evidence: list[dict] = []
    retrieval_plan = state["retrieval_plan"]
    try:
        for outcome in _execute_plan(retrieval_plan, state, kb_client):
            payload = outcome.payload
            evidence.extend(payload.results)
            trace.append(
                {
                    "node": "knowledge",
                    "step": "kb_search",
                    "request_index": outcome.index,
                    "query": outcome.query,
                    "entity_types": outcome.entity_types,
                    "top_k": outcome.top_k,
                    "count": len(payload.results),
                    "elapsed_ms": outcome.elapsed_ms,
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
        return {
            **state,
            "evidence": [],
            "error": {
                "code": "kb_unavailable",
                "message": "Knowledge Base is temporarily unavailable.",
                "retryable": True,
            },
            "trace": trace,
        }
