from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from loguru import logger

from src.core_ai.nextrip_agent.constants import (
    DEFAULT_TYPED_KB_VERSION,
    KbVersion,
    is_typed_kb_version,
)
from src.core_ai.nextrip_agent.schemas import KbSearchPayload, TypedKbPayload, TypedTarget
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

    def query_typed(
        self,
        *,
        query: str,
        top_k: int,
        kb_version: KbVersion = DEFAULT_TYPED_KB_VERSION,
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
    return [
        _run_search(
            index=index,
            request=request,
            state=state,
            kb_client=kb_client,
        )
        for index, request in enumerate(retrieval_plan, start=1)
    ]


def knowledge_node(state: NexTripAgentState, kb_client: SupportsKbSearch) -> NexTripAgentState:
    if is_typed_kb_version(state.get("kb_version")):
        return _knowledge_typed(state, kb_client)
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
            trace.extend(_knowledge_trace(payload.trace))
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


def _knowledge_typed(
    state: NexTripAgentState,
    kb_client: SupportsKbSearch,
) -> NexTripAgentState:
    trace = list(state.get("trace") or [])
    started_at = perf_counter()
    kb_version = state.get("kb_version") or DEFAULT_TYPED_KB_VERSION
    try:
        query = _query_with_city(state["message"], state.get("city"))
        payload = TypedKbPayload.model_validate(
            kb_client.query_typed(
                query=query,
                top_k=state["top_k"],
                kb_version=kb_version,
            )
        )
        if payload.error:
            trace.extend(_knowledge_trace(payload.trace))
            return {
                **state,
                "evidence": [],
                "facts": [],
                "query_plan": payload.query_plan,
                "error": payload.error.model_dump(),
                "trace": trace,
            }
        candidates = _typed_candidates(payload)
        evidence = _attach_candidate_sources(
            candidates,
            payload,
        )
        trace.extend(_knowledge_trace(payload.trace))
        trace.append(
            {
                "node": "knowledge",
                "status": "completed",
                "kb_version": kb_version,
                "query": query,
                "count": len(evidence),
                "fact_count": len(payload.facts),
                "elapsed_ms": int((perf_counter() - started_at) * 1000),
            }
        )
        return {
            **state,
            "answer_type": payload.answer_type,
            "evidence": evidence,
            "facts": payload.facts,
            "missing_fields": payload.missing_fields,
            "query_plan": payload.query_plan,
            "matched_paths": payload.matched_paths,
            "constraint_results": payload.constraint_results,
            "required_tools": payload.required_tools,
            "trace": trace,
        }
    except Exception as exc:
        logger.exception(
            "NexTrip node knowledge typed error session_id={} kb_version={} error_type={}",
            state.get("session_id") or "-",
            kb_version,
            exc.__class__.__name__,
        )
        trace.append(
            {
                "node": "knowledge",
                "status": "error",
                "kb_version": kb_version,
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
        )
        return {
            **state,
            "evidence": [],
            "facts": [],
            "query_plan": {},
            "matched_paths": [],
            "constraint_results": [],
            "required_tools": [],
            "error": {
                "code": "kb_unavailable",
                "message": f"Knowledge Base {kb_version.upper()} is temporarily unavailable.",
                "retryable": True,
            },
            "trace": trace,
        }


def _attach_candidate_sources(
    candidates: list[dict],
    payload: TypedKbPayload,
) -> list[dict]:
    sources_by_subject = {
        citation.subject_id: {
            "name": citation.source_name,
            "url": citation.url,
        }
        for citation in payload.evidence
        if citation.subject_id is not None
    }
    if len(candidates) == 1 and not sources_by_subject and payload.evidence:
        citation = payload.evidence[0]
        sources_by_subject[candidates[0]["place_id"]] = {
            "name": citation.source_name,
            "url": citation.url,
        }
    return [
        {
            **candidate,
            "source": sources_by_subject.get(candidate["place_id"], {}),
        }
        for candidate in candidates
    ]


def _typed_candidates(payload: TypedKbPayload) -> list[dict]:
    if payload.recommendations:
        return payload.recommendations
    if payload.entities:
        return payload.entities
    if payload.answer_type == "recommendation" and _has_named_concept_target(payload):
        return []
    return _target_candidates(payload.targets)


def _has_named_concept_target(payload: TypedKbPayload) -> bool:
    targets = payload.query_plan.get("targets")
    if not isinstance(targets, list):
        return False
    return any(
        isinstance(target, dict)
        and target.get("kind") in {"dish", "activity", "concept"}
        and bool(target.get("value"))
        for target in targets
    )


def _target_candidates(targets: list[TypedTarget]) -> list[dict]:
    return [
        {
            "place_id": target.target_id,
            "name": target.name,
            "entity_type": target.kind,
            "category": target.kind,
            "score": target.score,
            "description": target.description,
        }
        for target in targets
    ]


def _knowledge_trace(events: list[dict]) -> list[dict]:
    return [
        {**event, "node": event.get("node") or "knowledge"}
        for event in events
    ]
