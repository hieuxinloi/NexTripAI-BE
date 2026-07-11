from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx
from loguru import logger

from src.shared.logging import safe_text
from src.shared.request_context import current_request_id


class KbClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._client = client if client is not None else httpx.Client(trust_env=False)
        self._owns_client = client is None

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
        return self._post("/api/kb/search", payload, timeout=30.0)

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
        return self._post("/api/kb/answer", payload, timeout=60.0)

    def query_v2(self, *, query: str, top_k: int, kb_version: str = "v2") -> dict[str, Any]:
        return self._post(
            "/api/kb/query",
            {"query": query, "kb_version": kb_version, "top_k": top_k},
            timeout=60.0,
        )

    def _post(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
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
            response = self._client.post(
                f"{self.base_url}{path}",
                json=payload,
                headers={"X-Request-ID": current_request_id()},
                timeout=timeout,
            )
            response.raise_for_status()
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
            len(data.get("results") or data.get("entities") or data.get("recommendations") or []),
            int((perf_counter() - started_at) * 1000),
        )
        return data
