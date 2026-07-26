from __future__ import annotations

from loguru import logger

from src.core_ai.nextrip_agent.answer_generation import (
    SupportsAnswerGeneration,
    fact_value_text,
    facts_for_answer,
)
from src.core_ai.nextrip_agent.constants import DEFAULT_TYPED_KB_VERSION, is_typed_kb_version
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


def _display_missing_fields(fields: list[str]) -> list[str]:
    labels: list[str] = []
    for field in fields:
        if field == "query_constraints":
            labels.append("tiêu chí hoặc điều kiện ưu tiên")
        elif field.startswith("distance_between:"):
            _, origin, destination = field.split(":", 2)
            labels.append(f"khoảng cách giữa {origin} và {destination}")
        elif field == "city":
            labels.append("thành phố hoặc khu vực")
        else:
            labels.append(field)
    return labels


def answer_node(
    state: NexTripAgentState,
    answer_generator: SupportsAnswerGeneration | None = None,
) -> NexTripAgentState:
    trace = list(state.get("trace") or [])
    evidence = list(state.get("evidence") or [])
    facts = list(state.get("facts") or [])
    missing_fields = list(state.get("missing_fields") or [])
    required_tools = list(state.get("required_tools") or [])
    not_found_targets = [
        value
        for field in missing_fields
        if (value := _not_found_value(field)) is not None
    ]
    unverified_geo_scopes = [
        field.removeprefix("verified_geo_candidates:")
        for field in missing_fields
        if field.startswith("verified_geo_candidates:")
    ]
    if required_tools:
        answer = "Câu hỏi này cần dữ liệu động từ: " + ", ".join(required_tools) + "."
    elif not_found_targets:
        answer = (
            "Mình chưa tìm thấy "
            + ", ".join(not_found_targets)
            + f" trong Knowledge Base {str(state.get('kb_version') or '').upper()}."
        )
    elif unverified_geo_scopes:
        answer = (
            f"Knowledge Base {str(state.get('kb_version') or '').upper()} chưa có "
            "địa điểm với quan hệ vị trí đủ tin cậy tại "
            + ", ".join(unverified_geo_scopes)
            + ". Cần bổ sung địa chỉ hoặc GeoArea đã xác minh."
        )
    elif missing_fields:
        answer = _clarification_answer(missing_fields)
    elif is_typed_kb_version(state.get("kb_version")) and facts:
        answer = _format_typed_facts(evidence, facts, state.get("kb_version") or DEFAULT_TYPED_KB_VERSION)
    elif is_typed_kb_version(state.get("kb_version")) and state.get("answer_type") == "entity_detail":
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


def _not_found_value(field: str) -> str | None:
    parts = field.split(":", 2)
    if len(parts) != 3 or parts[0] != "not_found":
        return None
    return parts[2].strip() or None


def _clarification_answer(missing_fields: list[str]) -> str:
    if "query_constraints" in missing_fields:
        return (
            "Bạn muốn tìm ở thành phố nào và ưu tiên loại địa điểm hoặc "
            "trải nghiệm nào?"
        )
    unresolved_concepts = [
        field.removeprefix("concept:")
        for field in missing_fields
        if field.startswith("concept:")
    ]
    if unresolved_concepts:
        return (
            "Mình chưa hiểu rõ tiêu chí “"
            + ", ".join(unresolved_concepts)
            + "”. Bạn có thể mô tả cụ thể hơn không?"
        )
    labels = {
        "city": "thành phố",
        "travel_date": "ngày dự kiến",
        "entity_type": "loại địa điểm",
    }
    readable = [labels.get(field, field.replace("_", " ")) for field in missing_fields]
    return f"Bạn vui lòng bổ sung {', '.join(readable)}."


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

