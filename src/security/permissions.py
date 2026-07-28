from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, status

from .auth import Principal, current_principal


def require_roles(*allowed_roles: str) -> Callable[..., Principal]:
    allowed = frozenset(allowed_roles)

    async def dependency(
        principal: Principal = Depends(current_principal),
    ) -> Principal:
        if principal.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )
        return principal

    return dependency


require_admin = require_roles("admin")
require_support_or_admin = require_roles("support", "admin")
