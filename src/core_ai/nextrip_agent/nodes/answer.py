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
    if not evidence:
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

