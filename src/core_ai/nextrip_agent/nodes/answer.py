from __future__ import annotations

from loguru import logger

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


def answer_node(state: NexTripAgentState) -> NexTripAgentState:
    trace = list(state.get("trace") or [])
    evidence = list(state.get("evidence") or [])
    facts = list(state.get("facts") or [])
    missing_fields = list(state.get("missing_fields") or [])
    if missing_fields:
        answer = f"Mình cần bạn bổ sung: {', '.join(missing_fields)}."
    elif state.get("kb_version") == "v2" and facts:
        answer = _format_typed_facts(evidence, facts, state.get("kb_version") or "v2")
    elif state.get("kb_version") == "v2" and state.get("answer_type") == "entity_detail":
        answer = "Không tìm thấy địa điểm khớp đủ tin cậy trong Knowledge Base V2."
    elif not evidence:
        answer = (
            "Hien tai chua lay duoc ket qua tu Knowledge Base. "
            "Hay kiem tra KB service, Neo4j va Gemini/Vertex AI config roi thu lai."
        )
    else:
        lines = ["Minh tim duoc mot so goi y tu Knowledge Base:"]
        for index, item in enumerate(evidence, start=1):
            lines.append(_format_evidence_line(index, item))
        answer = "\n".join(lines)
    trace.append({"node": "answer", "status": "completed"})
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
        lines.append(f"- {label}: {fact.get('value')}{unit}")
    return "\n".join(lines)

