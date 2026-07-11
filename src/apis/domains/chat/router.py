from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse
from src.apis.domains.chat.service import KnowledgeBaseUnavailableError, handle_chat
from src.infra.kb_client import KbClient

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    kb_client: KbClient = http_request.app.state.kb_client
    try:
        return handle_chat(request, kb_client)
    except KnowledgeBaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
