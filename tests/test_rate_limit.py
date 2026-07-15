from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.security.rate_limit import InMemoryRateLimiter


def test_rate_limiter_rejects_requests_over_window_limit() -> None:
    limiter = InMemoryRateLimiter(requests=2, window_seconds=60)
    limiter.check("user")
    limiter.check("user")

    with pytest.raises(HTTPException) as error:
        limiter.check("user")

    assert error.value.status_code == 429
    assert "Retry-After" in error.value.headers
