from __future__ import annotations

from fastapi import APIRouter

from src.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "nextrip-be",
        "kb_base_url": settings().nextrip_kb_base_url,
    }
