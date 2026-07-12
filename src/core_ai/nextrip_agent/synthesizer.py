from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.core_ai.nextrip_agent.answer_generation import (
    SupportsAnswerGeneration,
    facts_for_answer,
)
from src.core_ai.nextrip_agent.constants import KbVersion
from src.core_ai.nextrip_agent.nodes.answer import answer_node
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.weather import WeatherAssessment


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    trace: dict[str, Any]
    unresolved_tools: list[str]


def synthesize_answer(
    *,
    question: str,
    kb_version: KbVersion,
    graph: AgentResult,
    graph_used: bool,
    weather: WeatherAssessment | None,
    weather_requested: bool,
    weather_trace: dict[str, Any],
    answer_generator: SupportsAnswerGeneration | None,
) -> SynthesisResult:
    unresolved_tools = list(graph.required_tools)
    if weather is not None:
        unresolved_tools = [tool for tool in unresolved_tools if tool != "weather"]
    elif weather_requested and weather_trace.get("status") == "unavailable":
        if "weather" not in unresolved_tools:
            unresolved_tools.append("weather")

    can_use_llm = (
        answer_generator is not None
        and not graph.missing_fields
        and not unresolved_tools
        and bool(graph.evidence or graph.facts or weather)
    )
    if can_use_llm:
        try:
            synthesize = getattr(answer_generator, "synthesize", None)
            if callable(synthesize):
                answer = synthesize(
                    question=question,
                    answer_type=graph.answer_type,
                    evidence=graph.evidence,
                    facts=facts_for_answer(graph.facts),
                    matched_paths=graph.matched_paths,
                    weather=weather.model_dump(mode="json") if weather else None,
                )
                return SynthesisResult(
                    answer=answer,
                    unresolved_tools=unresolved_tools,
                    trace={
                        "node": "answer_synthesizer",
                        "status": "completed",
                        "generator": "llm_grounded_combined",
                        "sources": _source_names(graph_used, weather),
                    },
                )
            if weather is None:
                answer = answer_generator.generate(
                    question=question,
                    answer_type=graph.answer_type,
                    evidence=graph.evidence,
                    facts=facts_for_answer(graph.facts),
                    matched_paths=graph.matched_paths,
                )
                return SynthesisResult(
                    answer=answer,
                    unresolved_tools=unresolved_tools,
                    trace={
                        "node": "answer_synthesizer",
                        "status": "completed",
                        "generator": "llm_grounded",
                        "sources": _source_names(graph_used, weather),
                    },
                )
        except Exception as exc:
            logger.exception(
                "NexTrip answer synthesis failed error_type={}",
                exc.__class__.__name__,
            )
            fallback_reason = exc.__class__.__name__
        else:
            fallback_reason = "synthesizer_not_supported"
    else:
        fallback_reason = "llm_not_available_or_context_incomplete"

    answer = _fallback_answer(
        question=question,
        kb_version=kb_version,
        graph=graph,
        graph_used=graph_used,
        weather=weather,
        weather_requested=weather_requested,
        weather_trace=weather_trace,
        unresolved_tools=unresolved_tools,
    )
    return SynthesisResult(
        answer=answer,
        unresolved_tools=unresolved_tools,
        trace={
            "node": "answer_synthesizer",
            "status": "fallback",
            "generator": "template_combined",
            "reason": fallback_reason,
            "sources": _source_names(graph_used, weather),
        },
    )


def format_weather_answer(weather: WeatherAssessment) -> str:
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


def _fallback_answer(
    *,
    question: str,
    kb_version: KbVersion,
    graph: AgentResult,
    graph_used: bool,
    weather: WeatherAssessment | None,
    weather_requested: bool,
    weather_trace: dict[str, Any],
    unresolved_tools: list[str],
) -> str:
    if weather_requested and weather_trace.get("status") == "needs_input":
        return "Bạn muốn xem thời tiết ở Quy Nhơn hay Đà Nẵng?"
    parts: list[str] = []
    if graph_used and not graph.error:
        rendered = answer_node(
            {
                "message": question,
                "kb_version": kb_version,
                "answer_type": graph.answer_type,
                "evidence": graph.evidence,
                "facts": graph.facts,
                "missing_fields": graph.missing_fields,
                "matched_paths": graph.matched_paths,
                "required_tools": unresolved_tools,
                "trace": [],
            },
            None,
        )
        graph_answer = str(rendered.get("answer") or "").strip()
        if graph_answer:
            parts.append(graph_answer)
    if weather is not None:
        parts.append(format_weather_answer(weather))
    elif weather_requested and weather_trace.get("status") == "unavailable":
        parts.append("Mình chưa thể lấy dữ liệu thời tiết lúc này.")
    if not parts:
        return graph.answer or "Mình chưa có đủ dữ liệu để trả lời câu hỏi này."
    return "\n\n".join(parts)


def _source_names(graph_used: bool, weather: WeatherAssessment | None) -> list[str]:
    sources = []
    if graph_used:
        sources.append("graphrag")
    if weather is not None:
        sources.append("weather")
    return sources
