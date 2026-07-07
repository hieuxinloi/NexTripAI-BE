from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_TOP_K_PER_ENTITY = 10

ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "cafe": ("cafe", "ca phe", "coffee"),
    "restaurant": ("nha hang", "quan an", "restaurant"),
    "hotel": ("khach san", "hotel"),
    "attraction": ("diem tham quan", "dia diem", "attraction"),
    "nightlife": ("bar", "pub", "club", "nightlife"),
}

ENTITY_QUERY_TEXT: dict[str, str] = {
    "cafe": "cafe",
    "restaurant": "nha hang",
    "hotel": "khach san",
    "attraction": "dia diem tham quan",
    "nightlife": "bar pub club",
}


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    entity_types: list[str] | None
    top_k: int


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.replace("đ", "d").replace("Đ", "d").lower()


def _clamp_top_k(value: str) -> int:
    return max(1, min(int(value), MAX_TOP_K_PER_ENTITY))


def _entity_count_matches(message: str) -> list[tuple[int, str, int]]:
    matches: list[tuple[int, str, int]] = []
    normalized_message = normalize_text(message)

    for entity_type, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            pattern = rf"\b(\d+)\s+(?:quan\s+)?{re.escape(alias)}\b"
            match = re.search(pattern, normalized_message)
            if match:
                matches.append((match.start(), entity_type, _clamp_top_k(match.group(1))))
                break

    return sorted(matches, key=lambda item: item[0])


def build_retrieval_plan(
    *,
    message: str,
    entity_types: list[str] | None,
    top_k: int,
) -> list[RetrievalRequest]:
    if entity_types:
        return [RetrievalRequest(query=message, entity_types=entity_types, top_k=top_k)]

    typed_counts = _entity_count_matches(message)
    if typed_counts:
        return [
            RetrievalRequest(
                query=ENTITY_QUERY_TEXT[entity_type],
                entity_types=[entity_type],
                top_k=count,
            )
            for _, entity_type, count in typed_counts
        ]

    return [RetrievalRequest(query=message, entity_types=None, top_k=top_k)]
