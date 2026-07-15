from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from src.apis.domains.chat.schemas import ChatHistoryResponse
from src.infra.chat_store import ChatStore
from src.security.auth import Principal, current_principal


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


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
