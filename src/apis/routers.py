from __future__ import annotations

from fastapi import APIRouter, FastAPI

from src.apis.domains.chat.router import router as chat_router
from src.apis.domains.health.router import router as health_router
from src.apis.domains.sessions.router import router as sessions_router

API_ROUTERS: tuple[APIRouter, ...] = (
    health_router,
    chat_router,
    sessions_router,
)


def include_api_routers(app: FastAPI) -> None:
    for api_router in API_ROUTERS:
        app.include_router(api_router)
