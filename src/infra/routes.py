from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol

import httpx

from src.infra.resilience import CircuitBreaker, TtlCache, retry


ROUTES_ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_TRAVEL_MODES = {
    "car": "DRIVE",
    "motorbike": "TWO_WHEELER",
    "bicycle": "BICYCLE",
    "walking": "WALK",
}


class RouteUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RouteWaypoint:
    latitude: float
    longitude: float
    place_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class RouteResult:
    origin: RouteWaypoint
    destination: RouteWaypoint
    travel_mode: str
    distance_meters: int
    duration_seconds: int
    duration_source: Literal["google_route", "user_speed"]
    speed_kmh: float | None
    provider: Literal["google_routes"] = "google_routes"


class SupportsRoutes(Protocol):
    @property
    def configured(self) -> bool: ...

    def compute_route(
        self,
        origin: RouteWaypoint,
        destination: RouteWaypoint,
        *,
        travel_mode: str,
        speed_kmh: float | None = None,
    ) -> RouteResult: ...


@dataclass(frozen=True)
class _CachedRoute:
    distance_meters: int
    duration_seconds: int


class GoogleRoutesClient:
    def __init__(
        self,
        api_key: str | None,
        *,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 604_800,
        client: httpx.Client | None = None,
        retry_attempts: int = 2,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._cache_ttl_seconds = cache_ttl_seconds
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds, trust_env=False)
        self._retry_attempts = retry_attempts
        self._circuit = CircuitBreaker(
            circuit_failure_threshold,
            circuit_recovery_seconds,
        )
        self._cache: TtlCache[_CachedRoute] = TtlCache()

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def compute_route(
        self,
        origin: RouteWaypoint,
        destination: RouteWaypoint,
        *,
        travel_mode: str,
        speed_kmh: float | None = None,
    ) -> RouteResult:
        if not self.configured:
            raise RouteUnavailable("Google Routes API is not configured.")
        google_mode = GOOGLE_TRAVEL_MODES.get(travel_mode)
        if google_mode is None:
            raise ValueError(f"Unsupported route travel mode: {travel_mode}")
        if speed_kmh is not None and not 0 < speed_kmh <= 200:
            raise ValueError("speed_kmh must be between 0 and 200")

        cache_key = _route_cache_key(origin, destination, google_mode)
        route = self._cache.get(cache_key)
        if route is None:
            route = self._request_route(origin, destination, google_mode)
            self._cache.set(cache_key, route, self._cache_ttl_seconds)

        duration_seconds = route.duration_seconds
        duration_source: Literal["google_route", "user_speed"] = "google_route"
        if speed_kmh is not None:
            duration_seconds = round(
                route.distance_meters / 1000 / speed_kmh * 3600
            )
            duration_source = "user_speed"
        return RouteResult(
            origin=origin,
            destination=destination,
            travel_mode=travel_mode,
            distance_meters=route.distance_meters,
            duration_seconds=duration_seconds,
            duration_source=duration_source,
            speed_kmh=speed_kmh,
        )

    def _request_route(
        self,
        origin: RouteWaypoint,
        destination: RouteWaypoint,
        google_mode: str,
    ) -> _CachedRoute:
        def operation() -> httpx.Response:
            response = self.client.post(
                ROUTES_ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": self._api_key,
                    "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
                },
                json={
                    "origin": _waypoint_payload(origin),
                    "destination": _waypoint_payload(destination),
                    "travelMode": google_mode,
                    "computeAlternativeRoutes": False,
                },
            )
            response.raise_for_status()
            return response

        try:
            response = self._circuit.call(
                lambda: retry(
                    operation,
                    attempts=self._retry_attempts,
                    retryable=_is_retryable_route_error,
                )
            )
            routes = response.json().get("routes") or []
            if not routes:
                raise RouteUnavailable("Google Routes returned no route.")
            distance_meters = int(routes[0]["distanceMeters"])
            duration_seconds = _duration_seconds(str(routes[0]["duration"]))
            if distance_meters < 0 or duration_seconds < 0:
                raise ValueError("Route distance and duration must be non-negative")
            return _CachedRoute(distance_meters, duration_seconds)
        except RouteUnavailable:
            raise
        except Exception as exc:
            raise RouteUnavailable("Google Routes request failed.") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def _waypoint_payload(waypoint: RouteWaypoint) -> dict:
    return {
        "location": {
            "latLng": {
                "latitude": waypoint.latitude,
                "longitude": waypoint.longitude,
            }
        }
    }


def _route_cache_key(
    origin: RouteWaypoint,
    destination: RouteWaypoint,
    google_mode: str,
) -> str:
    return ":".join(
        (
            google_mode,
            f"{origin.latitude:.5f}",
            f"{origin.longitude:.5f}",
            f"{destination.latitude:.5f}",
            f"{destination.longitude:.5f}",
        )
    )


def _duration_seconds(value: str) -> int:
    if not value.endswith("s"):
        raise ValueError("Google route duration must use seconds")
    try:
        return round(Decimal(value[:-1]))
    except InvalidOperation as exc:
        raise ValueError("Invalid Google route duration") from exc


def _is_retryable_route_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))
