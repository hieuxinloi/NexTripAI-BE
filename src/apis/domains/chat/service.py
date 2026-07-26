from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from loguru import logger

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse, EvidenceItem
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
from src.core_ai.nextrip_agent.constants import is_typed_kb_version
from src.core_ai.nextrip_agent.conversation import (
    SupportsConversationContextualization,
    answer_memory_context,
    resolve_conversation_context,
    resolve_turn,
)
from src.core_ai.nextrip_agent.orchestrator import TravelOrchestrator
from src.core_ai.nextrip_agent.synthesizer import synthesize_answer
from src.infra.kb_client import KbClient
from src.infra.chat_store import ChatStore
from src.infra.weather import OpenMeteoWeatherClient

DEFAULT_TOP_K = 5
TYPED_QUERY_RESULT_CEILING = 20


class KnowledgeBaseUnavailableError(RuntimeError):
    pass


def resolve_top_k(request: ChatRequest) -> int:
    if request.top_k is not None:
        return request.top_k
    if is_typed_kb_version(request.kb_version):
        return TYPED_QUERY_RESULT_CEILING
    return DEFAULT_TOP_K


def handle_chat(
    request: ChatRequest,
    kb_client: KbClient,
    answer_generator: SupportsAnswerGeneration | None = None,
    *,
    weather_client: OpenMeteoWeatherClient | None = None,
    chat_store: ChatStore | None = None,
    chat_history_limit: int = 8,
    conversation_contextualizer: SupportsConversationContextualization | None = None,
) -> ChatResponse:
    started_at = perf_counter()
    user_id = getattr(request, "user_id", None)
    top_k = resolve_top_k(request)
    user_message_id = uuid4().hex
    assistant_message_id = uuid4().hex
    history = _recent_messages(
        chat_store,
        request.session_id,
        chat_history_limit,
        user_id=user_id,
    )
    session_memory = _get_session_memory(
        chat_store,
        request.session_id,
        user_id=user_id,
    )
    context = resolve_conversation_context(
        message=request.message,
        explicit_city=request.city,
        history=history,
        explicit_travel_date=request.travel_date,
    )
    resolved_turn = resolve_turn(
        message=request.message,
        history=history,
        context=context,
        contextualizer=conversation_contextualizer,
        prior_summary=str(session_memory.get("summary") or "") or None,
    )
    graph_message = resolved_turn.resolution.standalone_message
    _save_message(
        chat_store,
        request,
        user_message_id,
        "user",
        request.message,
        city=context.city,
        metadata={
            "context_city_source": context.city_source,
            "travel_date": request.travel_date.isoformat()
            if request.travel_date
            else None,
            "conversation_route": resolved_turn.resolution.route,
        },
    )
    logger.info(
        "Chat turn start session_id={} city={} entity_types={} top_k={} message_len={}",
        request.session_id,
        context.city or "-",
        request.entity_types or [],
        top_k,
        len(request.message),
    )
    if resolved_turn.resolution.route == "conversation":
        answer = str(resolved_turn.resolution.direct_answer or "").strip()
        trace = [context.trace_event(), resolved_turn.trace_event()]
        resolved_context = _resolved_context(context.to_dict(), resolved_turn)
        response = ChatResponse(
            session_id=request.session_id,
            message_id=assistant_message_id,
            answer=answer,
            intent="conversation_memory",
            orchestration_mode="conversation",
            resolved_context=resolved_context,
            kb_version=request.kb_version,
            trace=trace,
        )
        _save_message(
            chat_store,
            request,
            assistant_message_id,
            "assistant",
            answer,
            city=context.city,
            metadata={
                "kb_version": request.kb_version,
                "resolved_context": resolved_context,
                "trace": trace,
            },
        )
        _save_summary(
            chat_store,
            request.session_id,
            resolved_turn.resolution.summary,
            user_id=user_id,
        )
        logger.info(
            "Chat turn end session_id={} route=conversation answer_len={} elapsed_ms={}",
            request.session_id,
            len(answer),
            int((perf_counter() - started_at) * 1000),
        )
        return response

    orchestration = TravelOrchestrator(kb_client, weather_client).run(
        message=graph_message,
        session_id=request.session_id,
        city=context.city,
        entity_types=request.entity_types,
        top_k=top_k,
        kb_version=request.kb_version,
        travel_date=request.travel_date or context.travel_date,
        include_weather=request.include_weather,
        latitude=request.latitude,
        longitude=request.longitude,
    )
    agent_result = orchestration.graph
    weather = orchestration.weather
    if agent_result.error and weather is None:
        raise KnowledgeBaseUnavailableError(agent_result.error["message"])
    evidence = [EvidenceItem.model_validate(item) for item in agent_result.evidence]
    synthesis = synthesize_answer(
        question=request.message,
        kb_version=request.kb_version,
        graph=agent_result,
        graph_used=orchestration.plan.run_graph,
        weather=weather,
        weather_requested=orchestration.plan.run_weather,
        weather_trace=orchestration.weather_trace,
        answer_generator=answer_generator,
        conversation_context=answer_memory_context(
            history=history,
            resolved_turn=resolved_turn,
        ),
    )
    answer = synthesis.answer
    trace = [
        context.trace_event(),
        resolved_turn.trace_event(),
        *orchestration.trace,
        *agent_result.trace,
        orchestration.weather_trace,
        synthesis.trace,
    ]
    logger.info(
        "Chat turn end session_id={} evidence_count={} result_ids={} answer_len={} elapsed_ms={}",
        request.session_id,
        len(evidence),
        [item.place_id for item in evidence],
        len(answer),
        int((perf_counter() - started_at) * 1000),
    )
    resolved_context = _resolved_context(context.to_dict(), resolved_turn)
    response = ChatResponse(
        session_id=request.session_id,
        message_id=assistant_message_id,
        answer=answer,
        intent=agent_result.answer_type,
        orchestration_mode=orchestration.plan.mode.value,
        resolved_context=resolved_context,
        kb_version=request.kb_version,
        facts=agent_result.facts,
        evidence=evidence,
        recommendations=evidence
        if agent_result.answer_type == "recommendation"
        else [],
        missing_fields=agent_result.missing_fields,
        trace=trace,
        query_plan=agent_result.query_plan,
        matched_paths=agent_result.matched_paths,
        constraint_results=agent_result.constraint_results,
        required_tools=synthesis.unresolved_tools,
        weather=weather,
    )
    _save_message(
        chat_store,
        request,
        assistant_message_id,
        "assistant",
        answer,
        city=context.city,
        metadata={
            "kb_version": request.kb_version,
            "place_ids": [item.place_id for item in evidence],
            "referenced_entities": _referenced_entities(evidence),
            "weather_suitability": weather.suitability if weather else None,
            "travel_date": (request.travel_date or context.travel_date).isoformat()
            if (request.travel_date or context.travel_date)
            else None,
            "resolved_context": resolved_context,
            "trace": trace,
        },
    )
    _save_summary(
        chat_store,
        request.session_id,
        resolved_turn.resolution.summary,
        user_id=user_id,
    )
    return response


def _resolved_context(context: dict, resolved_turn) -> dict:
    return context | {
        "conversation_route": resolved_turn.resolution.route,
        "contextualized": (
            resolved_turn.resolution.standalone_message.strip()
            != resolved_turn.original_message.strip()
        ),
    }


def _referenced_entities(evidence: list[EvidenceItem]) -> list[dict[str, str]]:
    return [
        {
            "id": item.place_id,
            "name": str(item.name),
            "kind": item.entity_type or item.category or "place",
        }
        for item in evidence
        if item.name
    ][:20]


def _save_message(
    chat_store: ChatStore | None,
    request: ChatRequest,
    message_id: str,
    role: str,
    content: str,
    city: str | None,
    metadata: dict | None = None,
) -> None:
    if chat_store is None:
        return
    try:
        chat_store.save_message(
            request.session_id,
            message_id,
            role,
            content,
            user_id=getattr(request, "user_id", None),
            city=city,
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception(
            "Chat persistence failed session_id={} role={} error_type={}",
            request.session_id,
            role,
            exc.__class__.__name__,
        )


def _recent_messages(
    chat_store: ChatStore | None,
    session_id: str,
    limit: int,
    *,
    user_id: str | None = None,
) -> list[dict]:
    if chat_store is None:
        return []
    try:
        return chat_store.recent_messages(session_id, limit, user_id=user_id)
    except Exception as exc:
        logger.exception(
            "Chat history load failed session_id={} error_type={}",
            session_id,
            exc.__class__.__name__,
        )
        return []


def _get_session_memory(
    chat_store: ChatStore | None,
    session_id: str,
    *,
    user_id: str | None = None,
) -> dict:
    if chat_store is None:
        return {}
    try:
        return chat_store.get_session_memory(session_id, user_id=user_id)
    except Exception as exc:
        logger.exception(
            "Conversation memory load failed session_id={} error_type={}",
            session_id,
            exc.__class__.__name__,
        )
        return {}


def _save_summary(
    chat_store: ChatStore | None,
    session_id: str,
    summary: str | None,
    *,
    user_id: str | None = None,
) -> None:
    if chat_store is None or not summary:
        return
    try:
        chat_store.save_session_memory(
            session_id,
            {"summary": summary},
            user_id=user_id,
        )
    except Exception as exc:
        logger.exception(
            "Conversation memory save failed session_id={} error_type={}",
            session_id,
            exc.__class__.__name__,
        )
