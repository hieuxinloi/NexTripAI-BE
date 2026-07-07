from __future__ import annotations

from pydantic import BaseModel


class AgentRuntimeConfig(BaseModel):
    top_k: int = 5
    use_kb_answer: bool = False
