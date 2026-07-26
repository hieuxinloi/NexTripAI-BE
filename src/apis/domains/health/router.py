from __future__ import annotations

from functools import partial

from anyio import to_thread
from fastapi import APIRouter, Request, Response

from src.apis.domains.chat.worker_pool import ChatWorkerPool
from src.config import settings
from src.infra.kb_client import KbClient

router = APIRouter(tags=["health"])


def _version_sort_key(version: str) -> int:
    try:
        return int(version.removeprefix("v"))
    except ValueError:
        return 0


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok", "service": "nextrip-be"}


@router.get("/api/kb/versions")
async def kb_versions(request: Request) -> dict[str, object]:
    app_settings = request.app.state.settings
    kb_client: KbClient = request.app.state.kb_client
    try:
        dependency = await to_thread.run_sync(
            kb_client.readiness,
            abandon_on_cancel=True,
        )
        ready_versions = sorted(
            {str(item) for item in dependency.get("ready_versions") or []},
            key=_version_sort_key,
            reverse=True,
        )
    except Exception as exc:
        return {
            "status": "not_ready",
            "default_version": None,
            "versions": [],
            "error": exc.__class__.__name__,
        }
    default_version = (
        app_settings.active_kb_version
        if app_settings.active_kb_version in ready_versions
        else next(iter(ready_versions), None)
    )
    return {
        "status": "ready" if ready_versions else "not_ready",
        "default_version": default_version,
        "versions": [
            {
                "kb_version": version,
                "label": version.upper(),
            }
            for version in ready_versions
        ],
    }


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, object]:
    app_settings = request.app.state.settings
    kb_client: KbClient = request.app.state.kb_client
    try:
        dependency = await to_thread.run_sync(
            partial(kb_client.readiness, force=True),
            abandon_on_cancel=True,
        )
        ready_versions = set(dependency.get("ready_versions") or [])
        candidates = [
            app_settings.active_kb_version,
            *app_settings.kb_fallback_version_list,
        ]
        selected = next((item for item in candidates if item in ready_versions), None)
    except Exception as exc:
        dependency = {"status": "not_ready", "error": exc.__class__.__name__}
        selected = None
    if selected is None:
        response.status_code = 503
    return {
        "status": "ready" if selected else "not_ready",
        "service": "nextrip-be",
        "selected_kb_version": selected,
        "dependencies": {"kb": dependency},
    }


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    worker_pool: ChatWorkerPool = request.app.state.chat_worker_pool
    return {
        "status": "ok",
        "service": "nextrip-be",
        "kb_base_url": settings().nextrip_kb_base_url,
        "weather": "ready",
        "weather_provider": "open-meteo",
        "chat_store": request.app.state.chat_store.backend_name,
        "answer_generation": (
            "gemini_ai_studio"
            if request.app.state.answer_generator is not None
            else "template"
        ),
        "conversation_memory": (
            "gemini_contextualizer"
            if request.app.state.conversation_contextualizer is not None
            else "structured_context_only"
        ),
        "gemini_context_model": request.app.state.settings.gemini_context_model,
        "gemini_answer_model": request.app.state.settings.gemini_answer_model,
        "gemini_thinking_level": request.app.state.settings.gemini_thinking_level,
        "evaluation": {
            "available": request.app.state.evaluation_manager.available,
            "judge_model": request.app.state.settings.gemini_context_model,
            "persistence": request.app.state.evaluation_manager.persistence_backend,
        },
        "auth_mode": request.app.state.settings.auth_mode,
        "active_kb_version": request.app.state.settings.active_kb_version,
        "worker_pool": worker_pool.statistics(),
    }
