from __future__ import annotations

from fastapi import APIRouter, FastAPI

from src.apis.domains.auth.router import router as auth_router
from src.apis.domains.chat.router import router as chat_router
from src.apis.domains.evaluations.router import router as evaluations_router
from src.apis.domains.health.router import router as health_router
from src.apis.domains.sessions.router import router as sessions_router
from src.apis.domains.trips.router import router as trips_router
from src.apis.domains.preferences.router import router as preferences_router
from src.apis.domains.admin.router import router as admin_router

API_ROUTERS: tuple[APIRouter, ...] = (
    health_router,
    auth_router,
    chat_router,
    evaluations_router,
    sessions_router,
    trips_router,
    preferences_router,
    admin_router,
)


def include_api_routers(app: FastAPI) -> None:
    for api_router in API_ROUTERS:
        app.include_router(api_router)
