from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from src.infra.current_data_client import CurrentDataClient


def test_current_data_client_uses_private_contract_and_default_occupancy() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"results": []})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = CurrentDataClient(
        "http://current:8020/",
        "internal-key",
        client=http,
        retry_attempts=1,
    )

    client.hotel_availability(
        hotel_ids=["hotel_qn_001", "hotel_qn_002"],
        check_in=date(2026, 8, 31),
        stay_nights=1,
    )
    client.recommend_transport(
        origin_id="hotel_qn_001",
        destination_id="attr_qn_001",
        departure_time=datetime(2026, 8, 31, 8, tzinfo=timezone.utc),
    )

    hotel_request, traffic_request = requests
    assert hotel_request.url.path == "/api/current/hotel-availability/search"
    assert hotel_request.headers["X-NexTrip-Current-Key"] == "internal-key"
    hotel_payload = __import__("json").loads(hotel_request.content)
    assert hotel_payload["occupancy"] == {"adults": 2, "children": 0, "rooms": 1}
    assert hotel_payload["refresh_if_missing"] is False
    assert hotel_payload["include_stale"] is True
    assert traffic_request.url.path == "/api/current/traffic/recommendations"


def test_single_hotel_can_refresh_on_demand() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"results": []})

    client = CurrentDataClient(
        "http://current:8020",
        "key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_attempts=1,
    )
    client.hotel_availability(
        hotel_ids=["hotel_qn_001"],
        check_in=date(2026, 8, 31),
        stay_nights=2,
    )

    assert captured["refresh_if_missing"] is True
    assert captured["include_stale"] is True
    assert captured["stay_nights"] == 2
