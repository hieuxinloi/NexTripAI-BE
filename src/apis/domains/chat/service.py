from __future__ import annotations

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse, EvidenceItem
from src.config import settings
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.infra.kb_client import KbClient

DEFAULT_TOP_K = 5
MAX_INFERRED_TOP_K = 10


def infer_top_k(message: str) -> int:
    for token in message.split():
        if token.isdigit():
            return max(1, min(int(token), MAX_INFERRED_TOP_K))
    return DEFAULT_TOP_K


def handle_chat(request: ChatRequest) -> ChatResponse:
    kb_client = KbClient(settings().nextrip_kb_base_url)
    agent_result = run_nextrip_agent(
        message=request.message,
        session_id=request.session_id,
        city=request.city,
        entity_types=request.entity_types,
        top_k=request.top_k if request.top_k is not None else infer_top_k(request.message),
        kb_client=kb_client,
    )
    evidence = [EvidenceItem.model_validate(item) for item in agent_result.evidence]
    return ChatResponse(
        answer=agent_result.answer,
        evidence=evidence,
        recommendations=evidence,
        trace=agent_result.trace,
    )
