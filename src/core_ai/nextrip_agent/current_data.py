from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from src.core_ai.nextrip_agent.schemas import AgentResult


_VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


class SupportsCurrentData(Protocol):
    def places(self, place_ids: list[str]) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...

    def recommend_transport(
        self,
        *,
        origin_id: str,
        destination_id: str,
        departure_time: datetime,
    ) -> dict[str, Any]: ...


def enrich_current_data(
    graph: AgentResult,
    client: SupportsCurrentData | None,
    *,
    travel_date: date | None,
) -> tuple[AgentResult, dict[str, Any]]:
    """Attach operational data without changing the public response shape.

    Current place fields are stored below ``evidence[].attributes.current``;
    hotel windows below ``attributes.hotel_availability``; and each computed
    route below ``itinerary[].slots[].transport_to_next``. All additions are
    optional and every remote failure preserves the original graph result.
    """

    if client is None:
        return graph, {"node": "current_data", "status": "disabled"}

    evidence = [dict(item) for item in graph.evidence]
    itinerary = _copy_itinerary(graph.itinerary)
    place_ids = _canonical_place_ids(
        [item.get("place_id") for item in evidence]
        + [
            slot.get("place_id")
            for day_item in itinerary
            for slot in day_item.get("slots", [])
        ]
    )
    warnings = list(graph.warnings)
    failures: list[str] = []
    completed: list[str] = []

    current_by_id: dict[str, dict[str, Any]] = {}
    if place_ids:
        try:
            current_by_id = _current_places_by_id(client.places(place_ids))
            evidence = [
                _merge_current_place(item, current_by_id) for item in evidence
            ]
            completed.append("places")
        except Exception as exc:
            failures.append(f"places:{exc.__class__.__name__}")

    hotel_ids = _hotel_ids(evidence, itinerary)
    hotel_by_id: dict[str, dict[str, Any]] = {}
    if hotel_ids and travel_date is not None:
        try:
            response = client.hotel_availability(
                hotel_ids=hotel_ids,
                check_in=travel_date,
                stay_nights=_stay_nights(graph.query_plan),
            )
            hotel_by_id = {
                str(item.get("hotel_id")): item
                for item in response.get("results", [])
                if isinstance(item, dict) and item.get("hotel_id")
            }
            evidence = [
                _merge_hotel_availability(item, hotel_by_id) for item in evidence
            ]
            itinerary = _merge_hotel_slots(itinerary, hotel_by_id)
            completed.append("hotel_availability")
        except Exception as exc:
            failures.append(f"hotel_availability:{exc.__class__.__name__}")

    route_count = 0
    route_failures = 0
    if itinerary and travel_date is not None:
        itinerary, route_count, route_failures = _attach_transport(
            itinerary,
            client,
            travel_date,
        )
        if route_count:
            completed.append("traffic")
        if route_failures:
            failures.append(f"traffic:{route_failures}")

    required_tools = _remaining_tools(
        graph.required_tools,
        completed=completed,
        route_failures=route_failures,
    )
    if failures and "current_data_partial" not in warnings:
        warnings.append("current_data_partial")
    status = "completed" if completed and not failures else (
        "partial" if completed else "unavailable"
    )
    updated = graph.model_copy(
        update={
            "evidence": evidence,
            "itinerary": itinerary,
            "required_tools": required_tools,
            "warnings": warnings,
        }
    )
    return updated, {
        "node": "current_data",
        "status": status,
        "completed": completed,
        "failures": failures,
        "place_count": len(current_by_id),
        "hotel_count": len(hotel_by_id),
        "route_count": route_count,
    }


def _canonical_place_ids(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        place_id = value.strip()
        if place_id.startswith("v8:"):
            place_id = place_id[3:]
        if place_id not in result:
            result.append(place_id)
    return result


def _current_places_by_id(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for item in payload.get("items", []):
        if not isinstance(item, dict) or item.get("status") == "missing":
            continue
        envelope = item.get("current")
        if not isinstance(envelope, dict):
            continue
        place = envelope.get("place")
        if not isinstance(place, dict) or not place.get("place_id"):
            continue
        current[str(place["place_id"])] = envelope
    return current


def _merge_current_place(
    evidence: dict[str, Any],
    current_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    place_id = str(evidence.get("place_id") or "")
    envelope = current_by_id.get(place_id)
    if envelope is None:
        return evidence
    place = envelope.get("place")
    if not isinstance(place, dict):
        return evidence
    attributes = dict(evidence.get("attributes") or {})
    attributes["current"] = {
        key: value
        for key, value in {
            "business_status": place.get("business_status"),
            "opening": place.get("opening"),
            "weekly_opening": place.get("weekly_opening"),
            "opening_hours": place.get("opening_hours"),
            "price_level": place.get("price_level"),
            "updated_at": place.get("updated_at"),
            "stale_after": place.get("stale_after"),
            "stale": envelope.get("stale"),
        }.items()
        if value is not None
    }
    location = place.get("location")
    if isinstance(location, dict):
        attributes.setdefault("latitude", location.get("lat"))
        attributes.setdefault("longitude", location.get("lng"))
    return {
        **evidence,
        "name": place.get("name") or evidence.get("name"),
        "city": place.get("city") or evidence.get("city"),
        "entity_type": place.get("entity_type") or evidence.get("entity_type"),
        "category": place.get("category") or evidence.get("category"),
        "attributes": attributes,
    }


def _hotel_ids(
    evidence: list[dict[str, Any]],
    itinerary: list[dict[str, Any]],
) -> list[str]:
    values = [
        item.get("place_id")
        for item in evidence
        if item.get("entity_type") == "hotel"
    ]
    values.extend(
        slot.get("place_id")
        for day_item in itinerary
        for slot in day_item.get("slots", [])
        if slot.get("entity_type") == "hotel"
    )
    return _canonical_place_ids(values)


def _stay_nights(query_plan: Mapping[str, Any]) -> int:
    duration = query_plan.get("duration_days")
    if isinstance(duration, int) and not isinstance(duration, bool):
        return max(1, min(duration - 1, 30))
    return 1


def _merge_hotel_availability(
    evidence: dict[str, Any],
    hotel_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    result = hotel_by_id.get(str(evidence.get("place_id") or ""))
    if result is None:
        return evidence
    attributes = dict(evidence.get("attributes") or {})
    attributes["hotel_availability"] = result
    return {**evidence, "attributes": attributes}


def _merge_hotel_slots(
    itinerary: list[dict[str, Any]],
    hotel_by_id: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for day_item in itinerary:
        for slot in day_item.get("slots", []):
            result = hotel_by_id.get(str(slot.get("place_id") or ""))
            if result is not None:
                slot["hotel_availability"] = result
    return itinerary


def _copy_itinerary(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **day_item,
            "slots": [dict(slot) for slot in day_item.get("slots", [])],
        }
        for day_item in value
    ]


def _attach_transport(
    itinerary: list[dict[str, Any]],
    client: SupportsCurrentData,
    start_date: date,
) -> tuple[list[dict[str, Any]], int, int]:
    jobs: list[tuple[dict[str, Any], str, str, datetime]] = []
    for day_item in itinerary:
        day_number = int(day_item.get("day") or 1)
        slots = day_item.get("slots", [])
        for origin, destination in zip(slots, slots[1:]):
            origin_id = str(origin.get("place_id") or "")
            destination_id = str(destination.get("place_id") or "")
            if not origin_id or not destination_id or origin_id == destination_id:
                continue
            departure = _departure_at(
                start_date + timedelta(days=max(day_number - 1, 0)),
                str(origin.get("end_time") or "00:00"),
            )
            jobs.append((origin, origin_id, destination_id, departure))

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(jobs)))) as executor:
        future_jobs = {
            executor.submit(
                client.recommend_transport,
                origin_id=origin_id,
                destination_id=destination_id,
                departure_time=departure,
            ): (slot, origin_id, destination_id, departure)
            for slot, origin_id, destination_id, departure in jobs
        }
        for future in as_completed(future_jobs):
            slot, origin_id, destination_id, departure = future_jobs[future]
            try:
                payload = future.result()
                slot["transport_to_next"] = _transport_summary(
                    payload,
                    origin_id=origin_id,
                    destination_id=destination_id,
                    departure=departure,
                )
                completed += 1
            except Exception as exc:
                slot["transport_to_next"] = {
                    "status": "unavailable",
                    "origin_place_id": origin_id,
                    "destination_place_id": destination_id,
                    "departure_time": departure.isoformat(),
                    "error_code": exc.__class__.__name__,
                }
                failed += 1
    return itinerary, completed, failed


def _departure_at(local_date: date, value: str) -> datetime:
    try:
        parsed = time.fromisoformat(value)
    except ValueError:
        parsed = time(0, 0)
    return datetime.combine(local_date, parsed, tzinfo=_VIETNAM_TIMEZONE)


def _transport_summary(
    payload: Mapping[str, Any],
    *,
    origin_id: str,
    destination_id: str,
    departure: datetime,
) -> dict[str, Any]:
    options = payload.get("options")
    normalized_options = options if isinstance(options, list) else []
    selected = next(
        (
            option
            for option in normalized_options
            if isinstance(option, dict) and option.get("recommended") is True
        ),
        None,
    )
    route_response = selected.get("route") if isinstance(selected, dict) else None
    route = (
        route_response.get("route")
        if isinstance(route_response, dict)
        and isinstance(route_response.get("route"), dict)
        else {}
    )
    return {
        "status": str(payload.get("status") or "unavailable"),
        "origin_place_id": origin_id,
        "destination_place_id": destination_id,
        "departure_time": departure.isoformat(),
        "recommended_mode": payload.get("recommended_mode"),
        "distance_meters": (
            selected.get("distance_meters") if isinstance(selected, dict) else None
        ),
        "duration_seconds": (
            selected.get("duration_seconds") if isinstance(selected, dict) else None
        ),
        "provider": route.get("provider"),
        "traffic_basis": route.get("traffic_basis"),
        "traffic_aware": route.get("traffic_aware"),
        "degraded": bool(payload.get("degraded")),
        "partial": bool(payload.get("partial")),
        "selection_reason": payload.get("selection_reason"),
        "alternatives": [
            {
                key: option.get(key)
                for key in (
                    "mode",
                    "status",
                    "distance_meters",
                    "duration_seconds",
                    "rank",
                    "recommended",
                    "reason_codes",
                )
            }
            for option in normalized_options
            if isinstance(option, dict)
        ],
    }


def _remaining_tools(
    tools: list[str],
    *,
    completed: list[str],
    route_failures: int,
) -> list[str]:
    satisfied: set[str] = set()
    if "places" in completed:
        satisfied.add("live_status")
    if "hotel_availability" in completed:
        satisfied.update({"booking", "live_price"})
    if "traffic" in completed and route_failures == 0:
        satisfied.update({"route", "transport"})
    return [tool for tool in tools if tool not in satisfied]


__all__ = ["SupportsCurrentData", "enrich_current_data"]
