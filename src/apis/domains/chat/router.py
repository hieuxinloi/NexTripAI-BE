from __future__ import annotations

from fastapi import APIRouter

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse
from src.apis.domains.chat.service import handle_chat

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat(request)
