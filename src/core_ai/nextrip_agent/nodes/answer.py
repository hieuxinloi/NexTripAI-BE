from __future__ import annotations

from loguru import logger

from src.core_ai.nextrip_agent.answer_generation import (
    SupportsAnswerGeneration,
    fact_value_text,
    facts_for_answer,
)
from src.core_ai.nextrip_agent.constants import DEFAULT_TYPED_KB_VERSION, TYPED_KB_VERSIONS
from src.core_ai.nextrip_agent.state import NexTripAgentState


def _display_value(item: dict, keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return str(value)
    return default


def _format_evidence_line(index: int, item: dict) -> str:
    name = _display_value(item, ("name", "place_id"), "khong ro ten")
    city = _display_value(item, ("city",), "khong ro thanh pho")
    category = _display_value(item, ("category", "entity_type"), "dia diem")
    return f"{index}. {name} - {category}, {city}."


def answer_node(
    state: NexTripAgentState,
    answer_generator: SupportsAnswerGeneration | None = None,
) -> NexTripAgentState:
    trace = list(state.get("trace") or [])
    evidence = list(state.get("evidence") or [])
    facts = list(state.get("facts") or [])
    missing_fields = list(state.get("missing_fields") or [])
    required_tools = list(state.get("required_tools") or [])
    not_found_entities = [
        field.removeprefix("not_found:entity:")
        for field in missing_fields
        if field.startswith("not_found:entity:")
    ]
    if required_tools:
        answer = "Câu hỏi này cần dữ liệu động từ: " + ", ".join(required_tools) + "."
    elif not_found_entities:
        answer = (
            "Mình chưa tìm thấy "
            + ", ".join(not_found_entities)
            + f" trong Knowledge Base {str(state.get('kb_version') or '').upper()}."
        )
    elif missing_fields:
        answer = f"Mình cần bạn bổ sung: {', '.join(missing_fields)}."
    elif state.get("kb_version") in TYPED_KB_VERSIONS and facts:
        answer = _format_typed_facts(evidence, facts, state.get("kb_version") or DEFAULT_TYPED_KB_VERSION)
    elif state.get("kb_version") in TYPED_KB_VERSIONS and state.get("answer_type") == "entity_detail":
        answer = f"Không tìm thấy địa điểm khớp đủ tin cậy trong Knowledge Base {str(state.get('kb_version')).upper()}."
    elif state.get("answer_type") == "unsupported":
        answer = "Câu hỏi này chưa được GraphRAG hiện tại hỗ trợ."
    elif not evidence:
        answer = "Không tìm thấy địa điểm phù hợp với yêu cầu trong Knowledge Base hiện tại."
    else:
        lines = ["Minh tim duoc mot so goi y tu Knowledge Base:"]
        for index, item in enumerate(evidence, start=1):
            lines.append(_format_evidence_line(index, item))
        answer = "\n".join(lines)
    generation_mode = "template"
    if answer_generator is not None and (evidence or facts) and not missing_fields and not required_tools:
        try:
            answer = answer_generator.generate(
                question=state["message"],
                answer_type=state.get("answer_type") or "kb_retrieval",
                evidence=evidence,
                facts=facts_for_answer(facts),
                matched_paths=list(state.get("matched_paths") or []),
            )
            generation_mode = "llm_grounded"
        except Exception as exc:
            logger.exception(
                "NexTrip grounded answer generation failed session_id={} error_type={}",
                state.get("session_id") or "-",
                exc.__class__.__name__,
            )
            trace.append({
                "node": "answer",
                "step": "llm_generation",
                "status": "fallback",
                "error_type": exc.__class__.__name__,
            })
    trace.append({"node": "answer", "status": "completed", "generator": generation_mode})
    logger.info(
        "NexTrip node answer completed session_id={} evidence_count={} answer_len={}",
        state.get("session_id") or "-",
        len(evidence),
        len(answer),
    )
    return {**state, "answer": answer, "trace": trace}


def _format_typed_facts(evidence: list[dict], facts: list[dict], kb_version: str) -> str:
    count_facts = [fact for fact in facts if fact.get("predicate") == "count"]
    if count_facts:
        labels = {
            "attraction": "Điểm tham quan",
            "cafe": "Quán cafe",
            "hotel": "Khách sạn",
            "nightlife": "Địa điểm nightlife",
            "restaurant": "Nhà hàng",
        }
        lines = [f"Knowledge Base {kb_version.upper()} đếm được:"]
        for fact in count_facts:
            label = labels.get(fact.get("entity_type"), fact.get("entity_type") or "Địa điểm")
            lines.append(f"- {label}: {fact['value']}")
        return "\n".join(lines)

    name = _display_value(evidence[0], ("name", "place_id"), "Địa điểm") if evidence else "Địa điểm"
    labels = {
        "address": "Địa chỉ",
        "location": "Tọa độ",
        "city": "Thành phố",
        "description": "Thông tin nổi bật",
        "altitude": "Độ cao",
        "opening_hours": "Giờ mở cửa",
        "rating": "Đánh giá",
        "review_count": "Số lượt đánh giá",
    }
    lines = [name]
    has_address = any(fact.get("predicate") == "address" for fact in facts)
    for fact in facts:
        predicate = fact.get("predicate")
        if predicate == "location" and has_address:
            continue
        label = labels.get(predicate, str(predicate))
        unit = f" {fact['unit']}" if fact.get("unit") else ""
        lines.append(f"- {label}: {fact_value_text(fact.get('value'))}{unit}")
    return "\n".join(lines)

