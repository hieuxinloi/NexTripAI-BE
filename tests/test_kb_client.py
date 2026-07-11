from __future__ import annotations

import httpx

from src.infra.kb_client import KbClient


def test_kb_client_reuses_injected_connection_pool() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"strategy": "v1", "results": [], "trace": []})

    transport = httpx.MockTransport(handler)
    shared_http_client = httpx.Client(transport=transport)
    kb_client = KbClient("http://kb.test", client=shared_http_client)

    for _ in range(2):
        kb_client.search(query="cafe", city=None, entity_types=None, top_k=3)
    kb_client.close()

    assert request_count == 2
    assert not shared_http_client.is_closed
    shared_http_client.close()


def test_kb_client_calls_v2_query_contract() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"kb_version": "v2", "entities": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kb_client = KbClient("http://kb.test", client=client)
    kb_client.query_typed(query="Mỹ Khê ở đâu?", top_k=5)

    assert captured["path"] == "/api/kb/query"
    assert '"kb_version":"v2"' in captured["body"]
    client.close()


def test_kb_client_calls_v3_query_contract() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"kb_version": "v3", "entities": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kb_client = KbClient("http://kb.test", client=client)
    kb_client.query_typed(query="hotel 5 sao", top_k=5, kb_version="v3")

    assert '"kb_version":"v3"' in captured["body"]
    client.close()


def test_kb_client_calls_v4_query_contract() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"kb_version": "v4", "entities": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kb_client = KbClient("http://kb.test", client=client)
    kb_client.query_typed(query="seafood for families", top_k=5, kb_version="v4")

    assert '"kb_version":"v4"' in captured["body"]
    client.close()
