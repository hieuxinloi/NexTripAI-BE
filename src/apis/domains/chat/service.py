from __future__ import annotations

from time import perf_counter

from loguru import logger

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse, EvidenceItem
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
from src.core_ai.nextrip_agent.constants import TYPED_KB_VERSIONS
from src.infra.kb_client import KbClient

DEFAULT_TOP_K = 5
TYPED_QUERY_RESULT_CEILING = 20


class KnowledgeBaseUnavailableError(RuntimeError):
    pass


def resolve_top_k(request: ChatRequest) -> int:
    if request.top_k is not None:
        return request.top_k
    if request.kb_version in TYPED_KB_VERSIONS:
        return TYPED_QUERY_RESULT_CEILING
    return DEFAULT_TOP_K


def handle_chat(
    request: ChatRequest,
    kb_client: KbClient,
    answer_generator: SupportsAnswerGeneration | None = None,
) -> ChatResponse:
    started_at = perf_counter()
    top_k = resolve_top_k(request)
    logger.info(
        "Chat turn start session_id={} city={} entity_types={} top_k={} message_len={}",
        request.session_id,
        request.city or "-",
        request.entity_types or [],
        top_k,
        len(request.message),
    )
    agent_result = run_nextrip_agent(
        message=request.message,
        session_id=request.session_id,
        city=request.city,
        entity_types=request.entity_types,
        top_k=top_k,
        kb_client=kb_client,
        kb_version=request.kb_version,
        answer_generator=answer_generator,
    )
    if agent_result.error:
        raise KnowledgeBaseUnavailableError(agent_result.error["message"])
    evidence = [EvidenceItem.model_validate(item) for item in agent_result.evidence]
    logger.info(
        "Chat turn end session_id={} evidence_count={} result_ids={} answer_len={} elapsed_ms={}",
        request.session_id,
        len(evidence),
        [item.place_id for item in evidence],
        len(agent_result.answer),
        int((perf_counter() - started_at) * 1000),
    )
    return ChatResponse(
        answer=agent_result.answer,
        intent=agent_result.answer_type,
        kb_version=request.kb_version,
        facts=agent_result.facts,
        evidence=evidence,
        recommendations=evidence if agent_result.answer_type == "recommendation" else [],
        missing_fields=agent_result.missing_fields,
        trace=agent_result.trace,
        query_plan=agent_result.query_plan,
        matched_paths=agent_result.matched_paths,
        constraint_results=agent_result.constraint_results,
        required_tools=agent_result.required_tools,
    )
