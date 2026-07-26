from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from src.apis.domains.chat.schemas import (
    ChatHistoryResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionListResponse,
    SessionSummary,
    SessionUpdateRequest,
)
from src.infra.chat_store import ChatStore
from src.security.auth import Principal, current_principal


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionCreateResponse)
def create_session(
    request: Request,
    payload: SessionCreateRequest | None = None,
    principal: Principal = Depends(current_principal),
) -> SessionCreateResponse:
    store: ChatStore = request.app.state.chat_store
    session_id = uuid4().hex
    data = store.create_session(
        session_id,
        user_id=principal.uid,
        title=payload.title if payload else None,
    )
    return SessionCreateResponse.model_validate(data)


@router.get("", response_model=SessionListResponse)
def list_sessions(
    request: Request,
    principal: Principal = Depends(current_principal),
    limit: int = Query(default=50, ge=1, le=100),
) -> SessionListResponse:
    store: ChatStore = request.app.state.chat_store
    return SessionListResponse(
        sessions=[SessionSummary.model_validate(item) for item in store.list_sessions(
            user_id=principal.uid,
            limit=limit,
        )]
    )


@router.get("/{session_id}/messages", response_model=ChatHistoryResponse)
def history(
    request: Request,
    principal: Principal = Depends(current_principal),
    session_id: str = Path(pattern=r"^[A-Za-z0-9._:-]{1,128}$"),
    limit: int = Query(default=50, ge=1, le=200),
) -> ChatHistoryResponse:
    store: ChatStore = request.app.state.chat_store
    try:
        messages = store.recent_messages(session_id, limit, user_id=principal.uid)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Session access denied.") from exc
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.patch("/{session_id}", response_model=SessionSummary)
def rename_session(
    payload: SessionUpdateRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
    session_id: str = Path(pattern=r"^[A-Za-z0-9._:-]{1,128}$"),
) -> SessionSummary:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Session title cannot be empty.")
    store: ChatStore = request.app.state.chat_store
    try:
        session = store.rename_session(session_id, title, user_id=principal.uid)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Session access denied.") from exc
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionSummary.model_validate(session)


@router.delete("/{session_id}")
def delete_session(
    request: Request,
    principal: Principal = Depends(current_principal),
    session_id: str = Path(pattern=r"^[A-Za-z0-9._:-]{1,128}$"),
) -> dict[str, object]:
    store: ChatStore = request.app.state.chat_store
    try:
        deleted = store.delete_session(session_id, user_id=principal.uid)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Session access denied.") from exc
    return {"session_id": session_id, "deleted": deleted}
