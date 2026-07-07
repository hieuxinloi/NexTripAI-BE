from __future__ import annotations

from typing import Any

import httpx


class KbClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

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
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{self.base_url}/api/kb/search", json=payload)
            response.raise_for_status()
            return response.json()

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
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base_url}/api/kb/answer", json=payload)
            response.raise_for_status()
            return response.json()
