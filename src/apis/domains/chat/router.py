from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from anyio import fail_after
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from src.apis.domains.chat.schemas import AuthenticatedChatRequest, ChatRequest, ChatResponse
from src.apis.domains.chat.idempotency import IdempotencyCoordinator
from src.apis.domains.chat.service import (
    ConversationKnowledgeBaseVersionError,
    KnowledgeBaseUnavailableError,
    KnowledgeBaseVersionMismatchError,
)
from src.apis.domains.chat.worker_pool import ChatWorkerPool
from src.infra.kb_client import KbClient
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
from src.core_ai.nextrip_agent.synthesizer import AnswerGenerationUnavailableError
from src.config import Settings
from src.infra.chat_store import ChatStore
from src.security.auth import Principal, current_principal
from src.security.rate_limit import InMemoryRateLimiter

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(current_principal),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatResponse:
    prepared = _prepare_request(request, http_request, principal)
    return await _execute_chat(prepared, http_request, principal, idempotency_key)


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(current_principal),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> StreamingResponse:
    prepared = _prepare_request(request, http_request, principal)

    async def events() -> AsyncIterator[str]:
        yield _sse("accepted", {"session_id": prepared.session_id})
        task = asyncio.create_task(
            _execute_chat(prepared, http_request, principal, idempotency_key)
        )
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=10)
            if not done:
                yield _sse("heartbeat", {"status": "processing"})
        try:
            result = await task
        except HTTPException as exc:
            yield _sse("error", {"status": exc.status_code, "detail": exc.detail})
            return
        except Exception:
            logger.exception("Streaming chat failed session_id={}", prepared.session_id)
            yield _sse("error", {"status": 500, "detail": "Chat processing failed."})
            return
        yield _sse("result", result.model_dump(mode="json"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def _prepare_request(
    request: ChatRequest,
    http_request: Request,
    principal: Principal,
) -> AuthenticatedChatRequest:
    limiter: InMemoryRateLimiter = http_request.app.state.rate_limiter
    client_host = http_request.client.host if http_request.client else "unknown"
    limiter.check(f"{principal.uid}:{client_host}")
    version = select_kb_version(
        request.kb_version,
        http_request,
        client_selected="kb_version" in request.model_fields_set,
        principal=principal,
    )
    return AuthenticatedChatRequest.model_validate(
        request.model_dump() | {"user_id": principal.uid, "kb_version": version}
    )


def select_kb_version(
    requested: str,
    http_request: Request,
    *,
    client_selected: bool,
    principal: Principal,
) -> str:
    app_settings: Settings = http_request.app.state.settings
    kb_client: KbClient = http_request.app.state.kb_client
    if client_selected and not principal.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only admins can select a Knowledge Base version.",
        )
    if client_selected and not app_settings.allow_client_kb_version:
        raise HTTPException(
            status_code=409,
            detail="Knowledge Base version selection is disabled.",
        )
    if client_selected:
        candidates = [requested]
    else:
        candidates = list(dict.fromkeys([
            app_settings.active_kb_version,
            *app_settings.kb_fallback_version_list,
        ]))
    preferred = candidates[0]
    try:
        try:
            readiness = kb_client.readiness()
        except AttributeError:
            readiness = {"ready_versions": list(kb_client.ready_versions())}
        ready_versions = {
            str(version) for version in readiness.get("ready_versions") or []
        }
    except Exception as exc:
        logger.warning("KB readiness lookup failed error_type={}", exc.__class__.__name__)
        if app_settings.environment == "production":
            raise HTTPException(status_code=503, detail="Knowledge Base is not ready.") from exc
        return preferred
    active_version = str(readiness.get("active_version") or "")
    if not client_selected and active_version in ready_versions:
        return active_version
    for version in candidates:
        if version in ready_versions:
            return version
    raise HTTPException(
        status_code=503,
        detail=f"No configured Knowledge Base version is ready: {candidates}",
    )


async def _execute_chat(
    request: AuthenticatedChatRequest,
    http_request: Request,
    principal: Principal,
    idempotency_key: str | None,
) -> ChatResponse:
    key = _validate_idempotency_key(idempotency_key)
    store: ChatStore = http_request.app.state.chat_store
    if key:
        try:
            cached = store.get_idempotent_response(
                request.session_id,
                key,
                user_id=principal.uid,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Session access denied.") from exc
        if cached is not None:
            return ChatResponse.model_validate(cached)

    async def operation() -> ChatResponse:
        if key:
            cached_after_coordination = store.get_idempotent_response(
                request.session_id,
                key,
                user_id=principal.uid,
            )
            if cached_after_coordination is not None:
                return ChatResponse.model_validate(cached_after_coordination)
        result = await _run_worker(request, http_request)
        if key:
            store.save_idempotent_response(
                request.session_id,
                key,
                result.model_dump(mode="json"),
                user_id=principal.uid,
                ttl_seconds=http_request.app.state.settings.idempotency_ttl_seconds,
            )
        return result

    if not key:
        return await operation()
    coordinator: IdempotencyCoordinator = http_request.app.state.idempotency
    return await coordinator.run(request.session_id, key, operation)


async def _run_worker(request: AuthenticatedChatRequest, http_request: Request) -> ChatResponse:
    kb_client: KbClient = http_request.app.state.kb_client
    answer_generator: SupportsAnswerGeneration | None = http_request.app.state.answer_generator
    worker_pool: ChatWorkerPool = http_request.app.state.chat_worker_pool
    try:
        with fail_after(http_request.app.state.settings.chat_request_timeout_seconds):
            return await worker_pool.submit(request, kb_client, answer_generator)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Chat processing timed out.") from exc
    except KnowledgeBaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KnowledgeBaseVersionMismatchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ConversationKnowledgeBaseVersionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AnswerGenerationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _validate_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", key):
        raise HTTPException(status_code=400, detail="Invalid Idempotency-Key.")
    return key


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
