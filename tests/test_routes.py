from __future__ import annotations

import httpx
import pytest

from src.core_ai.nextrip_agent.orchestrator import TravelOrchestrator
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.infra.routes import (
    GoogleRoutesClient,
    RouteResult,
    RouteUnavailable,
    RouteWaypoint,
)


def test_google_routes_uses_actual_distance_and_recalculates_user_speed() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"routes": [{"distanceMeters": 8200, "duration": "960s"}]},
        )

    client = GoogleRoutesClient(
        "maps-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_attempts=1,
    )
    origin = RouteWaypoint(13.885, 109.29, "eo-gio", "Eo Gió")
    destination = RouteWaypoint(13.87, 109.28, "ky-co", "Kỳ Co")

    first = client.compute_route(
        origin,
        destination,
        travel_mode="motorbike",
        speed_kmh=30,
    )
    second = client.compute_route(
        origin,
        destination,
        travel_mode="motorbike",
        speed_kmh=40,
    )

    assert first.distance_meters == second.distance_meters == 8200
    assert first.duration_seconds == 984
    assert second.duration_seconds == 738
    assert first.duration_source == second.duration_source == "user_speed"
    assert len(requests) == 1
    assert '"travelMode":"TWO_WHEELER"' in requests[0].content.decode()


def test_google_routes_requires_a_dedicated_maps_key() -> None:
    client = GoogleRoutesClient(None)
    waypoint = RouteWaypoint(16.0, 108.0)

    with pytest.raises(RouteUnavailable, match="not configured"):
        client.compute_route(waypoint, waypoint, travel_mode="car")


class _RouteClient:
    configured = True

    def compute_route(
        self,
        origin: RouteWaypoint,
        destination: RouteWaypoint,
        *,
        travel_mode: str,
        speed_kmh: float | None = None,
    ) -> RouteResult:
        return RouteResult(
            origin=origin,
            destination=destination,
            travel_mode=travel_mode,
            distance_meters=8200,
            duration_seconds=984,
            duration_source="user_speed",
            speed_kmh=speed_kmh,
        )


def test_v8_route_tool_replaces_unresolved_tool_with_grounded_facts() -> None:
    graph = AgentResult(
        answer="",
        required_tools=["route"],
        route_context={
            "endpoints": [
                {
                    "place_id": "eo-gio",
                    "name": "Eo Gió",
                    "latitude": 13.885,
                    "longitude": 109.29,
                },
                {
                    "place_id": "ky-co",
                    "name": "Kỳ Co",
                    "latitude": 13.87,
                    "longitude": 109.28,
                },
            ],
            "options": {"travel_mode": "motorbike", "speed_kmh": 30},
        },
    )
    orchestrator = TravelOrchestrator(None, None, route_client=_RouteClient())

    updated, trace = orchestrator._run_route(
        graph,
        latitude=None,
        longitude=None,
    )

    assert trace["status"] == "completed"
    assert updated.required_tools == []
    assert [fact["predicate"] for fact in updated.facts] == [
        "actual_route_distance",
        "route_duration",
        "user_speed",
    ]
    assert updated.facts[0]["value"] == 8.2
    assert updated.facts[1]["value"] == 16
