from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AuthProfileResponse(BaseModel):
    uid: str
    email: str | None = None
    display_name: str | None = None
    role: Literal["user", "support", "admin"]
    auth_mode: str
    permissions: list[str]
