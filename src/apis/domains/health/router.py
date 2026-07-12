from __future__ import annotations

from fastapi import APIRouter, Request

from src.apis.domains.chat.worker_pool import ChatWorkerPool
from src.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    worker_pool: ChatWorkerPool = request.app.state.chat_worker_pool
    return {
        "status": "ok",
        "service": "nextrip-be",
        "kb_base_url": settings().nextrip_kb_base_url,
        "worker_pool": worker_pool.statistics(),
    }
