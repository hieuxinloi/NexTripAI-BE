from __future__ import annotations

from datetime import date, datetime
from time import perf_counter
from typing import Any

import httpx
from loguru import logger

from src.infra.resilience import CircuitBreaker, retry
from src.shared.logging import safe_text
from src.shared.request_context import current_request_id


class CurrentDataClient:
    """Private HTTP adapter for the NexTrip Current Data facade.

    The backend owns this integration boundary. Frontend responses continue to
    use the public ``ChatResponse`` contract and never expose service URLs or
    provider credentials.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        client: httpx.Client | None = None,
        *,
        timeout_seconds: float = 20.0,
        retry_attempts: int = 2,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client if client is not None else httpx.Client(
            trust_env=False,
            timeout=timeout_seconds,
        )
        self._owns_client = client is None
        self._timeout = timeout_seconds
        self._retry_attempts = retry_attempts
        self._circuit = CircuitBreaker(
            circuit_failure_threshold,
            circuit_recovery_seconds,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def places(self, place_ids: list[str]) -> dict[str, Any]:
        return self._post(
            "/api/current/places/batch",
            {"place_ids": list(dict.fromkeys(place_ids))},
        )

    def readiness(self) -> dict[str, Any]:
        response = self._client.get(
            f"{self.base_url}/ready",
            timeout=min(self._timeout, 5.0),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Current Data readiness must be a JSON object")
        return payload

    def trip_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/current/trip-context", payload)

    def hotel_availability(
        self,
        *,
        hotel_ids: list[str],
        check_in: date,
        stay_nights: int,
        adults: int = 2,
        children: int = 0,
        rooms: int = 1,
        children_ages: list[int] | None = None,
        lookahead_days: int = 1,
        currency: str = "VND",
        refresh_if_missing: bool = True,
    ) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(hotel_ids))
        # The Current Data service deliberately permits refresh-on-miss for one
        # hotel only. A batch remains a read of accepted scheduled observations.
        effective_refresh = refresh_if_missing and len(unique_ids) == 1
        return self._post(
            "/api/current/hotel-availability/search",
            {
                "hotel_ids": unique_ids,
                "check_in": check_in.isoformat(),
                "stay_nights": stay_nights,
                "lookahead_days": lookahead_days,
                "occupancy": {
                    "adults": adults,
                    "children": children,
                    "rooms": rooms,
                },
                "children_ages": children_ages or [],
                "currency": currency,
                # Current Data hides expired observations unless explicitly
                # requested. The BE answer contract allows a clearly marked
                # stale price when no fresh quote is available, so request
                # those observations at this integration boundary. The
                # service still prefers fresh data and preserves ``stale`` on
                # fallback offers; this does not alter the five-hour TTL.
                "include_stale": bool(refresh_if_missing),
                "refresh_if_missing": effective_refresh,
            },
        )

    def recommend_transport(
        self,
        *,
        origin_id: str,
        destination_id: str,
        departure_time: datetime,
    ) -> dict[str, Any]:
        return self._post(
            "/api/current/traffic/recommendations",
            {
                "origin_id": origin_id,
                "destination_id": destination_id,
                "departure_time": departure_time.isoformat(),
                "candidate_modes": ["walk", "bicycle", "two_wheeler", "drive"],
                "objective": "balanced",
                "include_baseline": False,
                "force_refresh": False,
                "allow_stale_on_error": True,
            },
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = perf_counter()

        def operation() -> httpx.Response:
            response = self._client.post(
                f"{self.base_url}{path}",
                json=payload,
                headers={
                    "X-NexTrip-Current-Key": self._api_key,
                    "X-Request-ID": current_request_id(),
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response

        try:
            response = self._circuit.call(
                lambda: retry(
                    operation,
                    attempts=self._retry_attempts,
                    retryable=_is_retryable,
                )
            )
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("Current Data response must be a JSON object")
            return payload
        except Exception as exc:
            status_code = None
            response_body = None
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
                response_body = safe_text(exc.response.text, 500)
            logger.warning(
                "Current Data request failed path={} error_type={} status_code={} "
                "response_body={} elapsed_ms={}",
                path,
                exc.__class__.__name__,
                status_code,
                response_body,
                int((perf_counter() - started_at) * 1000),
            )
            raise


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.ReadTimeout):
        return False
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {
        408,
        429,
        500,
        502,
        503,
        504,
    }


__all__ = ["CurrentDataClient"]
