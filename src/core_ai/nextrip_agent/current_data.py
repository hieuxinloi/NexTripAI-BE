from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
import math
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
    include_traffic: bool = True,
    adults: int = 2,
    children: int = 0,
    rooms: int = 1,
    force_hotel_refresh: bool = False,
) -> tuple[AgentResult, dict[str, Any]]:
    """Attach operational data without changing the public response shape.

    Current place fields are stored below ``evidence[].attributes.current``;
    hotel windows below ``attributes.hotel_availability``; and each computed
    route below ``itinerary[].slots[].transport_to_next``. A route explicitly
    requested by the user is attached to the origin evidence as
    ``attributes.transport_to_destination``. All additions are optional and
    every remote failure preserves the original graph result.
    """

    if client is None:
        if not force_hotel_refresh:
            return graph, {"node": "current_data", "status": "disabled"}
        return (
            graph.model_copy(
                update={
                    "evidence": _without_hotel_availability(
                        [dict(item) for item in graph.evidence]
                    ),
                    "itinerary": _without_itinerary_hotel_availability(
                        _copy_itinerary(graph.itinerary)
                    ),
                }
            ),
            {"node": "current_data", "status": "disabled"},
        )

    evidence = [dict(item) for item in graph.evidence]
    facts = [dict(item) for item in graph.facts]
    itinerary = _copy_itinerary(graph.itinerary)
    if force_hotel_refresh:
        evidence = _without_hotel_availability(evidence)
        itinerary = _without_itinerary_hotel_availability(itinerary)
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
    missing_current_ids = [
        place_id
        for place_id in place_ids
        if not _evidence_has_current(evidence, place_id)
    ]
    if missing_current_ids:
        try:
            current_by_id = _current_places_by_id(client.places(missing_current_ids))
            evidence = [
                _merge_current_place(
                    item,
                    current_by_id,
                    travel_date=travel_date,
                )
                for item in evidence
            ]
            completed.append("places")
        except Exception as exc:
            failures.append(f"places:{exc.__class__.__name__}")
    elif place_ids:
        completed.append("places")

    hotel_ids = _hotel_ids(evidence, itinerary)
    hotel_by_id = _existing_hotel_availability(evidence)
    missing_hotel_ids = [
        hotel_id for hotel_id in hotel_ids if hotel_id not in hotel_by_id
    ]
    selected_hotel_ids = _itinerary_hotel_ids(itinerary)
    unpriced_selected_hotel_ids = [
        hotel_id
        for hotel_id in selected_hotel_ids
        if not _hotel_result_has_price(hotel_by_id.get(hotel_id))
    ]
    lookup_hotel_ids = list(
        dict.fromkeys([*missing_hotel_ids, *unpriced_selected_hotel_ids])
    )
    if lookup_hotel_ids and travel_date is not None:
        try:
            response = client.hotel_availability(
                hotel_ids=lookup_hotel_ids,
                check_in=travel_date,
                stay_nights=_stay_nights(graph.query_plan),
                adults=adults,
                children=children,
                rooms=rooms,
            )
            hotel_by_id.update(
                {
                    str(item.get("hotel_id")): item
                    for item in response.get("results", [])
                    if isinstance(item, dict) and item.get("hotel_id")
                }
            )
            if len(lookup_hotel_ids) > 1:
                for selected_hotel_id in selected_hotel_ids:
                    if _hotel_result_has_price(hotel_by_id.get(selected_hotel_id)):
                        continue
                    try:
                        refreshed = client.hotel_availability(
                            hotel_ids=[selected_hotel_id],
                            check_in=travel_date,
                            stay_nights=_stay_nights(graph.query_plan),
                            adults=adults,
                            children=children,
                            rooms=rooms,
                        )
                    except Exception as exc:
                        failures.append(
                            "hotel_selected_refresh:"
                            f"{selected_hotel_id}:{exc.__class__.__name__}"
                        )
                        continue
                    hotel_by_id.update(
                        {
                            str(item.get("hotel_id")): item
                            for item in refreshed.get("results", [])
                            if isinstance(item, dict) and item.get("hotel_id")
                        }
                    )
            evidence = [
                _merge_hotel_availability(item, hotel_by_id) for item in evidence
            ]
            itinerary = _merge_hotel_slots(itinerary, hotel_by_id)
            completed.append("hotel_availability")
        except Exception as exc:
            failures.append(f"hotel_availability:{exc.__class__.__name__}")
    elif hotel_ids and travel_date is not None:
        itinerary = _merge_hotel_slots(itinerary, hotel_by_id)
        completed.append("hotel_availability")

    route_count = 0
    route_failures = 0
    route_requested = bool({"route", "transport"}.intersection(graph.required_tools))
    if include_traffic and route_requested and not itinerary and len(place_ids) >= 2:
        origin_id, destination_id = place_ids[:2]
        departure = _on_demand_departure(travel_date)
        try:
            payload = client.recommend_transport(
                origin_id=origin_id,
                destination_id=destination_id,
                departure_time=departure,
            )
            summary = _transport_summary(
                payload,
                origin_id=origin_id,
                destination_id=destination_id,
                departure=departure,
            )
            _require_usable_transport(summary)
            evidence = _merge_direct_transport(evidence, origin_id, summary)
            facts.append(_direct_transport_fact(summary))
            route_count = 1
            completed.append("traffic")
        except Exception as exc:
            route_failures = 1
            failures.append(f"traffic:{exc.__class__.__name__}")
    elif include_traffic and itinerary:
        routing_reference_date = travel_date or (
            datetime.now(_VIETNAM_TIMEZONE).date() + timedelta(days=1)
        )
        itinerary, route_count, route_failures = _attach_transport(
            itinerary,
            client,
            routing_reference_date,
        )
        if travel_date is None:
            _mark_assumed_route_times(itinerary)
            if "traffic_reference_time_assumed" not in warnings:
                warnings.append("traffic_reference_time_assumed")
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
    status = (
        "completed"
        if completed and not failures
        else ("partial" if completed else "unavailable")
    )
    updated = graph.model_copy(
        update={
            "evidence": evidence,
            "facts": facts,
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
        "traffic_enabled": include_traffic,
    }


def _evidence_has_current(evidence: list[dict[str, Any]], place_id: str) -> bool:
    return any(
        str(item.get("place_id") or "") == place_id
        and isinstance((item.get("attributes") or {}).get("current"), dict)
        for item in evidence
    )


def _existing_hotel_availability(
    evidence: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for item in evidence:
        availability = (item.get("attributes") or {}).get("hotel_availability")
        place_id = str(item.get("place_id") or "")
        if place_id and isinstance(availability, dict):
            values[place_id] = availability
    return values


def _without_hotel_availability(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in evidence:
        copied = dict(item)
        attributes = copied.get("attributes")
        if isinstance(attributes, Mapping):
            copied_attributes = dict(attributes)
            copied_attributes.pop("hotel_availability", None)
            copied["attributes"] = copied_attributes
        result.append(copied)
    return result


def _without_itinerary_hotel_availability(
    itinerary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = _copy_itinerary(itinerary)
    for day_item in result:
        for slot in day_item.get("slots", []):
            slot.pop("hotel_availability", None)
    return result


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
    *,
    travel_date: date | None,
) -> dict[str, Any]:
    place_id = str(evidence.get("place_id") or "")
    envelope = current_by_id.get(place_id)
    if envelope is None:
        return evidence
    place = envelope.get("place")
    if not isinstance(place, dict):
        return evidence
    attributes = dict(evidence.get("attributes") or {})
    current = {
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
    attributes["current"] = current
    location = place.get("location")
    if isinstance(location, dict):
        latitude = location.get("lat", location.get("latitude"))
        longitude = location.get("lng", location.get("longitude"))
        if isinstance(latitude, (int, float)) and not isinstance(latitude, bool):
            attributes["lat"] = latitude
            attributes["latitude"] = latitude
        if isinstance(longitude, (int, float)) and not isinstance(longitude, bool):
            attributes["lng"] = longitude
            attributes["longitude"] = longitude
    effective_hours = _effective_current_opening(place, travel_date)
    if effective_hours is not None:
        attributes["effective_opening"] = effective_hours
        intervals = effective_hours.get("intervals")
        if isinstance(intervals, list) and len(intervals) == 1:
            interval = intervals[0]
            if isinstance(interval, dict):
                if isinstance(interval.get("opens_at"), str):
                    attributes["opening_hours_open"] = interval["opens_at"]
                if isinstance(interval.get("closes_at"), str):
                    attributes["opening_hours_close"] = interval["closes_at"]
    return {
        **evidence,
        "name": place.get("name") or evidence.get("name"),
        "city": place.get("city") or evidence.get("city"),
        "entity_type": place.get("entity_type") or evidence.get("entity_type"),
        "category": place.get("category") or evidence.get("category"),
        "attributes": attributes,
    }


def _effective_current_opening(
    place: Mapping[str, Any],
    travel_date: date | None,
) -> dict[str, Any] | None:
    daily = place.get("opening")
    if isinstance(daily, Mapping):
        local_date = str(daily.get("local_date") or "")
        if travel_date is not None and local_date == travel_date.isoformat():
            return {
                "source": "daily_observation",
                "date": local_date,
                "status": daily.get("status"),
                "is_24_hours": bool(daily.get("is_24_hours")),
                "intervals": _opening_intervals(daily.get("opening_intervals")),
            }

    weekly = place.get("weekly_opening")
    if travel_date is not None and isinstance(weekly, Mapping):
        weekday = travel_date.strftime("%A").lower()
        days = weekly.get("days")
        if isinstance(days, list):
            selected = next(
                (
                    item
                    for item in days
                    if isinstance(item, Mapping)
                    and str(item.get("day") or "").lower() == weekday
                ),
                None,
            )
            if selected is not None:
                return {
                    "source": "weekly_schedule",
                    "date": travel_date.isoformat(),
                    "status": "closed_today"
                    if selected.get("closed")
                    else "open_today",
                    "is_24_hours": bool(selected.get("open_24_hours")),
                    "intervals": _opening_intervals(selected.get("intervals")),
                }

    baseline = place.get("opening_hours")
    if isinstance(baseline, Mapping):
        opens_at = baseline.get("opens_at")
        closes_at = baseline.get("closes_at")
        intervals = (
            [{"opens_at": opens_at, "closes_at": closes_at}]
            if isinstance(opens_at, str) and isinstance(closes_at, str)
            else []
        )
        if intervals:
            return {
                "source": "verified_baseline",
                "date": travel_date.isoformat() if travel_date is not None else None,
                "status": "open_today",
                "is_24_hours": False,
                "intervals": intervals,
            }
    return None


def _opening_intervals(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        opens_at = item.get("opens_at")
        closes_at = item.get("closes_at")
        if not isinstance(opens_at, str) or not isinstance(closes_at, str):
            continue
        result.append(
            {
                "opens_at": opens_at,
                "closes_at": closes_at,
                "closes_next_day": bool(item.get("closes_next_day")),
            }
        )
    return result


def _hotel_ids(
    evidence: list[dict[str, Any]],
    itinerary: list[dict[str, Any]],
) -> list[str]:
    values = [
        item.get("place_id") for item in evidence if item.get("entity_type") == "hotel"
    ]
    values.extend(
        slot.get("place_id")
        for day_item in itinerary
        for slot in day_item.get("slots", [])
        if slot.get("entity_type") == "hotel"
    )
    return _canonical_place_ids(values)


def _itinerary_hotel_ids(itinerary: list[dict[str, Any]]) -> list[str]:
    return _canonical_place_ids(
        [
            slot.get("place_id")
            for day_item in itinerary
            for slot in day_item.get("slots", [])
            if slot.get("entity_type") == "hotel"
        ]
    )


def _hotel_result_has_price(result: object) -> bool:
    if not isinstance(result, Mapping):
        return False
    selected_index = result.get("selected_window_index")
    windows = result.get("windows")
    if not isinstance(selected_index, int) or not isinstance(windows, list):
        return False
    if selected_index < 0 or selected_index >= len(windows):
        return False
    selected = windows[selected_index]
    if not isinstance(selected, Mapping):
        return False
    return any(
        isinstance(offer, Mapping)
        and any(
            offer.get(field) is not None
            for field in ("total_amount", "nightly_amount", "amount", "min_amount")
        )
        for offer in selected.get("offers", [])
    )


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
    reused = 0
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
            if _has_usable_transport(origin, destination_id, departure):
                reused += 1
                continue
            jobs.append((origin, origin_id, destination_id, departure))

    completed = reused
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


def _has_usable_transport(
    slot: Mapping[str, Any],
    destination_id: str,
    departure: datetime,
) -> bool:
    value = slot.get("transport_to_next")
    return bool(
        isinstance(value, Mapping)
        and value.get("destination_place_id") == destination_id
        and value.get("departure_time") == departure.isoformat()
        and value.get("status") in {"recommended", "available"}
        and value.get("recommended_mode")
        and isinstance(value.get("duration_seconds"), int)
        and value.get("duration_seconds") > 0
    )


def _departure_at(local_date: date, value: str) -> datetime:
    try:
        parsed = time.fromisoformat(value)
    except ValueError:
        parsed = time(0, 0)
    return datetime.combine(local_date, parsed, tzinfo=_VIETNAM_TIMEZONE)


def _on_demand_departure(travel_date: date | None) -> datetime:
    now = datetime.now(_VIETNAM_TIMEZONE)
    if travel_date is None or travel_date == now.date():
        return now
    return datetime.combine(travel_date, now.timetz(), tzinfo=_VIETNAM_TIMEZONE)


def _require_usable_transport(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("status") not in {"recommended", "available"}
        or not summary.get("recommended_mode")
        or summary.get("distance_meters") is None
        or summary.get("duration_seconds") is None
    ):
        raise ValueError("traffic provider returned no usable recommendation")


def _merge_direct_transport(
    evidence: list[dict[str, Any]],
    origin_id: str,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in evidence:
        if str(item.get("place_id") or "") != origin_id:
            result.append(item)
            continue
        attributes = dict(item.get("attributes") or {})
        attributes["transport_to_destination"] = dict(summary)
        result.append({**item, "attributes": attributes})
    return result


def _direct_transport_fact(summary: Mapping[str, Any]) -> dict[str, Any]:
    mode = _transport_mode_label(str(summary["recommended_mode"]))
    distance_km = float(summary["distance_meters"]) / 1000
    duration_minutes = max(1, round(float(summary["duration_seconds"]) / 60))
    provider = str(summary.get("provider") or "traffic provider").upper()
    traffic_basis = (
        "có xét giao thông hiện tại"
        if summary.get("traffic_aware") is True
        else "theo dữ liệu định tuyến hiện có"
    )
    value = (
        f"Phương tiện đề xuất: {mode}; quãng đường theo đường thực tế: "
        f"{distance_km:.1f} km; thời gian dự kiến: {duration_minutes} phút; "
        f"nguồn định tuyến: {provider}, {traffic_basis}"
    )
    return {
        "fact_id": (
            "dynamic-route:"
            f"{summary['origin_place_id']}:{summary['destination_place_id']}:"
            f"{summary['departure_time']}"
        ),
        "subject_id": str(summary["origin_place_id"]),
        "predicate": "route_recommendation",
        "value": value,
        "value_type": "string",
        "confidence": 1.0,
        "evidence_ids": [],
    }


def _transport_mode_label(mode: str) -> str:
    return {
        "walk": "đi bộ",
        "bicycle": "xe đạp",
        "two_wheeler": "xe máy",
        "drive": "ô tô/taxi",
    }.get(mode, mode)


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
    distance_meters = (
        selected.get("distance_meters") if isinstance(selected, dict) else None
    )
    duration_seconds = (
        selected.get("duration_seconds") if isinstance(selected, dict) else None
    )
    recommended_mode = payload.get("recommended_mode")
    return {
        "status": str(payload.get("status") or "unavailable"),
        "origin_place_id": origin_id,
        "destination_place_id": destination_id,
        "departure_time": departure.isoformat(),
        "recommended_mode": recommended_mode,
        "distance_meters": distance_meters,
        "duration_seconds": duration_seconds,
        "distance_km": round(float(distance_meters) / 1000, 1)
        if isinstance(distance_meters, (int, float))
        else None,
        "duration_minutes": max(1, math.ceil(float(duration_seconds) / 60))
        if isinstance(duration_seconds, (int, float)) and duration_seconds > 0
        else None,
        "mode_label": _transport_mode_label(str(recommended_mode)),
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


def attach_itinerary_transport(
    itinerary: list[dict[str, Any]],
    client: SupportsCurrentData | None,
    *,
    start_date: date | None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Public fail-soft route attachment used by mutable plan revisions."""

    copied = _copy_itinerary(itinerary)
    if client is None:
        return copied, 0, 0
    reference_date = start_date or (
        datetime.now(_VIETNAM_TIMEZONE).date() + timedelta(days=1)
    )
    result, completed, failed = _attach_transport(copied, client, reference_date)
    if start_date is None:
        _mark_assumed_route_times(result)
    return result, completed, failed


def _mark_assumed_route_times(itinerary: list[dict[str, Any]]) -> None:
    for day_item in itinerary:
        for slot in day_item.get("slots", []):
            route = slot.get("transport_to_next")
            if isinstance(route, dict):
                route["reference_time_assumed"] = True


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


__all__ = ["SupportsCurrentData", "attach_itinerary_transport", "enrich_current_data"]
