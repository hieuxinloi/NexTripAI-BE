from __future__ import annotations

from time import perf_counter

from loguru import logger

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse, EvidenceItem
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.infra.kb_client import KbClient

DEFAULT_TOP_K = 5
MAX_INFERRED_TOP_K = 10


class KnowledgeBaseUnavailableError(RuntimeError):
    pass


def infer_top_k(message: str) -> int:
    for token in message.split():
        if token.isdigit():
            return max(1, min(int(token), MAX_INFERRED_TOP_K))
    return DEFAULT_TOP_K


def handle_chat(request: ChatRequest, kb_client: KbClient) -> ChatResponse:
    started_at = perf_counter()
    top_k = request.top_k if request.top_k is not None else infer_top_k(request.message)
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
        evidence=evidence,
        recommendations=evidence,
        trace=agent_result.trace,
    )
