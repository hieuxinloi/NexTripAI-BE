from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from loguru import logger

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse, EvidenceItem
from src.core_ai.nextrip_agent.graph import run_nextrip_agent
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
from src.core_ai.nextrip_agent.constants import TYPED_KB_VERSIONS
from src.infra.kb_client import KbClient
from src.core_ai.nextrip_agent.weather import WeatherAgent, WeatherAssessment
from src.infra.chat_store import ChatStore
from src.infra.weather import GoogleWeatherClient, WeatherUnavailable

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
    *,
    weather_client: GoogleWeatherClient | None = None,
    chat_store: ChatStore | None = None,
) -> ChatResponse:
    started_at = perf_counter()
    top_k = resolve_top_k(request)
    user_message_id = uuid4().hex
    assistant_message_id = uuid4().hex
    _save_message(
        chat_store,
        request,
        user_message_id,
        "user",
        request.message,
    )
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
    weather, weather_trace = _run_weather(request, agent_result.required_tools, weather_client)
    if agent_result.error and weather is None:
        raise KnowledgeBaseUnavailableError(agent_result.error["message"])
    evidence = [EvidenceItem.model_validate(item) for item in agent_result.evidence]
    required_tools = list(agent_result.required_tools)
    answer = agent_result.answer
    if weather is not None:
        required_tools = [tool for tool in required_tools if tool != "weather"]
        weather_answer = _weather_answer(weather)
        if agent_result.required_tools and not evidence and not agent_result.facts:
            answer = weather_answer
        else:
            answer = f"{answer}\n\n{weather_answer}"
    trace = [*agent_result.trace, weather_trace]
    logger.info(
        "Chat turn end session_id={} evidence_count={} result_ids={} answer_len={} elapsed_ms={}",
        request.session_id,
        len(evidence),
        [item.place_id for item in evidence],
        len(answer),
        int((perf_counter() - started_at) * 1000),
    )
    response = ChatResponse(
        session_id=request.session_id,
        message_id=assistant_message_id,
        answer=answer,
        intent=agent_result.answer_type,
        kb_version=request.kb_version,
        facts=agent_result.facts,
        evidence=evidence,
        recommendations=evidence if agent_result.answer_type == "recommendation" else [],
        missing_fields=agent_result.missing_fields,
        trace=trace,
        query_plan=agent_result.query_plan,
        matched_paths=agent_result.matched_paths,
        constraint_results=agent_result.constraint_results,
        required_tools=required_tools,
        weather=weather,
    )
    _save_message(
        chat_store,
        request,
        assistant_message_id,
        "assistant",
        answer,
        metadata={
            "kb_version": request.kb_version,
            "place_ids": [item.place_id for item in evidence],
            "weather_suitability": weather.suitability if weather else None,
            "trace": trace,
        },
    )
    return response


def _run_weather(
    request: ChatRequest,
    required_tools: list[str],
    weather_client: GoogleWeatherClient | None,
) -> tuple[WeatherAssessment | None, dict]:
    should_run = WeatherAgent.should_run(
        message=request.message,
        travel_date=request.travel_date,
        include_weather=request.include_weather,
        required_tools=required_tools,
    )
    if not should_run:
        return None, {"node": "weather", "status": "skipped"}
    if weather_client is None:
        return None, {
            "node": "weather",
            "status": "unavailable",
            "reason": "Weather client is not configured.",
        }
    started_at = perf_counter()
    try:
        weather = WeatherAgent(weather_client).run(
            message=request.message,
            city=request.city,
            travel_date=request.travel_date,
            latitude=request.latitude,
            longitude=request.longitude,
        )
    except WeatherUnavailable as exc:
        return None, {
            "node": "weather",
            "status": "unavailable",
            "reason": str(exc),
            "elapsed_ms": int((perf_counter() - started_at) * 1000),
        }
    return weather, {
        "node": "weather",
        "status": "completed",
        "suitability": weather.suitability,
        "elapsed_ms": int((perf_counter() - started_at) * 1000),
    }


def _weather_answer(weather: WeatherAssessment) -> str:
    temperature = ""
    if weather.min_temperature_c is not None and weather.max_temperature_c is not None:
        temperature = (
            f", khoảng {weather.min_temperature_c:g}-{weather.max_temperature_c:g}°C"
        )
    rain = (
        f", khả năng mưa {weather.precipitation_probability}%"
        if weather.precipitation_probability is not None
        else ""
    )
    return (
        f"Thời tiết {weather.location} ngày {weather.forecast_date:%d/%m}: "
        f"{weather.condition}{temperature}{rain}. {weather.advice}"
    )


def _save_message(
    chat_store: ChatStore | None,
    request: ChatRequest,
    message_id: str,
    role: str,
    content: str,
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
            user_id=request.user_id,
            city=request.city,
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception(
            "Chat persistence failed session_id={} role={} error_type={}",
            request.session_id,
            role,
            exc.__class__.__name__,
        )
