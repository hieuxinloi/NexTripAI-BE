from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx
from loguru import logger

from src.shared.logging import safe_text
from src.shared.request_context import current_request_id
from src.infra.resilience import CircuitBreaker, TtlCache, retry
from src.core_ai.personalization.models import SavedPlacesResponse


class KbClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        *,
        timeout_seconds: float = 25.0,
        auth_mode: str = "none",
        auth_audience: str | None = None,
        retry_attempts: int = 2,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
        admin_api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = (
            client
            if client is not None
            else httpx.Client(
                trust_env=False,
                timeout=timeout_seconds,
            )
        )
        self._owns_client = client is None
        self._timeout = timeout_seconds
        self._auth_mode = auth_mode
        self._auth_audience = auth_audience or self.base_url
        self._admin_api_key = admin_api_key
        self._retry_attempts = retry_attempts
        self._circuit = CircuitBreaker(
            circuit_failure_threshold,
            circuit_recovery_seconds,
        )
        self._token_cache: TtlCache[str] = TtlCache()
        self._health_cache: TtlCache[dict[str, Any]] = TtlCache()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(
        self,
        *,
        query: str,
        city: str | None,
        entity_types: list[str] | None,
        top_k: int,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "city": city,
            "entity_types": entity_types,
            "top_k": top_k,
        }
        return self._post("/api/kb/search", payload)

    def answer(
        self,
        *,
        query: str,
        city: str | None,
        entity_types: list[str] | None,
        top_k: int,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "city": city,
            "entity_types": entity_types,
            "top_k": top_k,
        }
        return self._post("/api/kb/answer", payload)

    def query_typed(
        self,
        *,
        query: str,
        top_k: int,
        kb_version: str = "v2",
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "kb_version": kb_version,
            "top_k": top_k,
        }
        if conversation_context:
            payload["conversation_context"] = conversation_context
        return self._post(
            "/api/kb/query",
            payload,
        )

    def personalized_recommendations(
        self,
        *,
        seed_place_ids: list[str],
        preferred_concepts: list[str],
        excluded_concepts: list[str],
        excluded_place_ids: list[str],
        preferred_cities: list[str],
        limit: int,
        kb_version: str = "v8",
    ) -> SavedPlacesResponse:
        payload = self._post(
            "/api/kb/recommendations",
            {
                "kb_version": kb_version,
                "seed_place_ids": seed_place_ids,
                "preferred_concepts": preferred_concepts,
                "excluded_concepts": excluded_concepts,
                "excluded_place_ids": excluded_place_ids,
                "preferred_cities": preferred_cities,
                "limit": limit,
            },
        )
        return SavedPlacesResponse.model_validate(payload)

    def places_by_ids(
        self,
        place_ids: list[str],
        *,
        kb_version: str = "v8",
    ) -> SavedPlacesResponse:
        payload = self._post(
            "/api/kb/places/batch",
            {"kb_version": kb_version, "place_ids": place_ids},
        )
        return SavedPlacesResponse.model_validate(payload)

    def nearby(
        self,
        *,
        anchor_place_id: str,
        entity_types: list[str] | None,
        city: str | None,
        radius_km: float = 5.0,
        excluded_place_ids: list[str] | None = None,
        limit: int = 8,
        kb_version: str = "v8",
    ) -> dict[str, Any]:
        return self._post(
            "/api/kb/nearby",
            {
                "kb_version": kb_version,
                "anchor_place_id": anchor_place_id,
                "entity_types": entity_types or [],
                "city": city,
                "radius_km": radius_km,
                "excluded_place_ids": excluded_place_ids or [],
                "limit": limit,
            },
        )

    def admin_deployments(self) -> dict[str, Any]:
        return self._admin_request("GET", "/api/kb/admin/deployments")

    def admin_validate_deployment(self, version: str) -> dict[str, Any]:
        return self._admin_request(
            "POST",
            f"/api/kb/admin/deployments/{version}/validate",
        )

    def admin_activate_deployment(self, version: str) -> dict[str, Any]:
        return self._admin_request(
            "POST",
            f"/api/kb/admin/deployments/{version}/activate",
        )

    def admin_rollback_deployment(self) -> dict[str, Any]:
        return self._admin_request("POST", "/api/kb/admin/deployments/rollback")

    def _admin_request(self, method: str, path: str) -> dict[str, Any]:
        if not self._admin_api_key:
            raise RuntimeError("KB administration is not configured.")
        response = self._client.request(
            method,
            f"{self.base_url}{path}",
            headers={
                **self._headers(),
                "X-KB-Admin-Key": self._admin_api_key,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        self._health_cache.clear()
        return response.json()

    def readiness(self, *, force: bool = False) -> dict[str, Any]:
        if not force:
            cached = self._health_cache.get("ready")
            if cached is not None:
                return cached
        response = self._client.get(
            f"{self.base_url}/ready",
            headers=self._headers(),
            timeout=min(self._timeout, 5.0),
        )
        response.raise_for_status()
        data = response.json()
        self._health_cache.set("ready", data, 5.0)
        return data

    def ready_versions(self) -> set[str]:
        data = self.readiness()
        versions = data.get("ready_versions") or []
        return {str(version) for version in versions}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = perf_counter()
        logger.info(
            "KB client request start path={} query={!r} city={} entity_types={} top_k={}",
            path,
            safe_text(str(payload.get("query") or "")),
            payload.get("city") or "-",
            payload.get("entity_types") or [],
            payload.get("top_k"),
        )
        try:

            def operation() -> httpx.Response:
                response = self._client.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    headers=self._headers(),
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response

            response = self._circuit.call(
                lambda: retry(
                    operation,
                    attempts=self._retry_attempts,
                    retryable=_is_retryable,
                )
            )
            data = response.json()
        except Exception as exc:
            logger.exception(
                "KB client request error path={} error_type={} elapsed_ms={}",
                path,
                exc.__class__.__name__,
                int((perf_counter() - started_at) * 1000),
            )
            raise
        logger.info(
            "KB client request end path={} status={} strategy={} result_count={} elapsed_ms={}",
            path,
            response.status_code,
            data.get("strategy") or data.get("kb_version") or "-",
            _payload_result_count(data),
            int((perf_counter() - started_at) * 1000),
        )
        return data

    def _headers(self) -> dict[str, str]:
        headers = {"X-Request-ID": current_request_id()}
        if self._auth_mode == "none":
            return headers
        if self._auth_mode != "google_oidc":
            raise RuntimeError(f"Unsupported KB auth mode: {self._auth_mode}")
        token = self._token_cache.get(self._auth_audience)
        if token is None:
            try:
                from google.auth.transport.requests import Request
                from google.oauth2 import id_token
            except ImportError as exc:
                raise RuntimeError(
                    "KB_AUTH_MODE=google_oidc requires google-auth."
                ) from exc
            token = id_token.fetch_id_token(Request(), self._auth_audience)
            self._token_cache.set(self._auth_audience, token, 45 * 60)
        headers["Authorization"] = f"Bearer {token}"
        return headers


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


def _payload_result_count(payload: dict[str, Any]) -> int:
    for field in ("results", "entities", "recommendations", "items"):
        value = payload.get(field)
        if isinstance(value, list) and value:
            return len(value)
    return 0
