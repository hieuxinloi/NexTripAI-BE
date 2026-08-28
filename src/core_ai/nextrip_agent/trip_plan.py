from __future__ import annotations

import math
import re
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from src.core_ai.nextrip_agent.weather import normalize_text


class PlanOperation(StrEnum):
    NONE = "none"
    CREATE = "create"
    REPLACE_SLOT = "replace_slot"
    REMOVE_SLOT = "remove_slot"
    ADD_SLOT = "add_slot"
    MOVE_SLOT = "move_slot"
    UPDATE_TIME = "update_time"
    UPDATE_CONSTRAINTS = "update_constraints"
    REPLAN_DAY = "replan_day"
    REPLAN_ALL = "replan_all"
    SUGGEST_NEARBY = "suggest_nearby"
    QUERY_PLAN = "query_plan"


class PlanMutation(BaseModel):
    """A semantic edit request. It never contains provider-computed facts."""

    operation: PlanOperation = PlanOperation.NONE
    expected_revision: int | None = Field(default=None, ge=1)
    target_slot_id: str | None = None
    target_day: int | None = Field(default=None, ge=1, le=30)
    target_order: int | None = Field(default=None, ge=1, le=30)
    target_role: str | None = None
    target_period: (
        Literal["morning", "lunch", "afternoon", "dinner", "evening"] | None
    ) = None
    target_place_name: str | None = None
    replacement_query: str | None = None
    replacement_place_name: str | None = None
    entity_type: str | None = None
    anchor_place_name: str | None = None
    destination_day: int | None = Field(default=None, ge=1, le=30)
    destination_order: int | None = Field(default=None, ge=1, le=30)
    start_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    day_start_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    day_end_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    budget_vnd: int | None = Field(default=None, ge=0)
    adults: int | None = Field(default=None, ge=1, le=30)
    children: int | None = Field(default=None, ge=0, le=20)
    rooms: int | None = Field(default=None, ge=1, le=10)
    preserve_unaffected: bool = True

    @model_validator(mode="after")
    def normalize_empty_operation(self) -> "PlanMutation":
        if self.operation is PlanOperation.NONE:
            self.preserve_unaffected = True
        return self


class PlanChange(BaseModel):
    operation: PlanOperation
    previous_revision: int | None = None
    revision: int | None = None
    target_slot_id: str | None = None
    changed_slot_ids: list[str] = Field(default_factory=list)
    preserved_slot_ids: list[str] = Field(default_factory=list)
    message: str | None = None


class BudgetSummary(BaseModel):
    status: Literal["complete", "partial", "unavailable"] = "unavailable"
    currency: str = "VND"
    party_size: int = Field(default=2, ge=1)
    budget_amount: int | None = Field(default=None, ge=0)
    estimated_min: int | None = Field(default=None, ge=0)
    estimated_max: int | None = Field(default=None, ge=0)
    within_budget: bool | None = None
    remaining_min: int | None = None
    remaining_max: int | None = None
    priced_item_count: int = Field(default=0, ge=0)
    estimable_item_count: int = Field(default=0, ge=0)
    missing_place_ids: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)


class ActiveTripPlan(BaseModel):
    plan_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    status: Literal["complete", "partial", "draft"] = "draft"
    city: str
    start_date: date | None = None
    duration_days: int = Field(ge=1, le=30)
    constraints: dict[str, Any] = Field(default_factory=dict)
    itinerary: list[dict[str, Any]] = Field(default_factory=list)
    selected_places: list[dict[str, Any]] = Field(default_factory=list)
    budget_summary: BudgetSummary | None = None
    last_operation: PlanOperation = PlanOperation.CREATE
    created_at: datetime
    updated_at: datetime


def active_plan_from_value(value: object) -> ActiveTripPlan | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return ActiveTripPlan.model_validate(value)
    except (TypeError, ValueError):
        return None


def compact_active_plan_context(plan: ActiveTripPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "plan_id": plan.plan_id,
        "revision": plan.revision,
        "status": plan.status,
        "city": plan.city,
        "start_date": plan.start_date.isoformat() if plan.start_date else None,
        "duration_days": plan.duration_days,
        "constraints": plan.constraints,
        "days": [
            {
                "day": day_item.get("day"),
                "slots": [
                    {
                        key: slot.get(key)
                        for key in (
                            "slot_id",
                            "order",
                            "start_time",
                            "end_time",
                            "place_id",
                            "name",
                            "entity_type",
                            "role",
                        )
                    }
                    for slot in day_item.get("slots", [])
                    if isinstance(slot, Mapping)
                ],
            }
            for day_item in plan.itinerary
            if isinstance(day_item, Mapping)
        ],
    }


def resolve_plan_mutation(
    message: str,
    active_plan: ActiveTripPlan | None,
    model_mutation: PlanMutation | Mapping[str, Any] | None = None,
) -> PlanMutation:
    """Resolve an explicit plan action while keeping unrelated turns independent."""

    if isinstance(model_mutation, PlanMutation):
        resolved = model_mutation
    elif isinstance(model_mutation, Mapping):
        try:
            resolved = PlanMutation.model_validate(model_mutation)
        except ValueError:
            resolved = PlanMutation()
    else:
        resolved = PlanMutation()

    plain = " ".join(normalize_text(message).split())
    budget = extract_budget_vnd(message)
    occupancy = extract_occupancy(message)
    if resolved.operation is not PlanOperation.NONE:
        if (
            active_plan is not None
            and resolved.operation
            in {PlanOperation.REPLAN_ALL, PlanOperation.REPLAN_DAY}
            and _is_add_command(plain)
        ):
            # A model may over-generalize an open-ended add request as a full
            # replan. Preserve the explicit edit verb and let grounded
            # discovery choose an unused candidate.
            resolved = resolved.model_copy(
                update={
                    "operation": PlanOperation.ADD_SLOT,
                    "target_slot_id": None,
                    "target_place_name": None,
                    "preserve_unaffected": True,
                }
            )
        if (
            resolved.operation is PlanOperation.REPLACE_SLOT
            and (budget is not None or occupancy)
            and not any(
                (
                    resolved.target_slot_id,
                    resolved.target_place_name,
                    resolved.replacement_place_name,
                    resolved.target_order,
                )
            )
            and (
                active_plan is None
                or _mentioned_plan_place(normalize_text(message), active_plan) is None
            )
        ):
            resolved = resolved.model_copy(
                update={"operation": PlanOperation.UPDATE_CONSTRAINTS}
            )
        updates: dict[str, Any] = {}
        if resolved.budget_vnd is None and budget is not None:
            updates["budget_vnd"] = budget
        if resolved.expected_revision is None and active_plan is not None:
            updates["expected_revision"] = active_plan.revision
        for field_name, value in occupancy.items():
            if getattr(resolved, field_name) is None:
                updates[field_name] = value
        if active_plan is not None:
            parsed_day = _first_int(plain, r"\bngay\s+(\d{1,2})\b")
            parsed_order = _first_int(
                plain,
                r"\b(?:dia diem|diem|slot|muc)\s+(?:thu\s+)?(\d{1,2})\b",
            )
            parsed_name = _mentioned_plan_place(plain, active_plan)
            if resolved.target_place_name is None and parsed_name is not None:
                updates["target_place_name"] = parsed_name
            if (
                resolved.operation is not PlanOperation.MOVE_SLOT
                and resolved.target_day is None
                and parsed_day is not None
            ):
                updates["target_day"] = parsed_day
            if resolved.target_order is None and parsed_order is not None:
                updates["target_order"] = parsed_order
            slot_start, slot_end = _requested_slot_times(plain)
            if resolved.operation is PlanOperation.ADD_SLOT:
                if resolved.replacement_place_name is None:
                    updates["replacement_place_name"] = _requested_added_place(
                        plain,
                        parsed_name,
                    )
                if resolved.start_time is None and slot_start is not None:
                    updates["start_time"] = slot_start
                if resolved.end_time is None and slot_end is not None:
                    updates["end_time"] = slot_end
            elif resolved.operation is PlanOperation.MOVE_SLOT:
                destination_day = _first_int(
                    plain,
                    r"\b(?:sang|qua|vao)\s+ngay\s+(\d{1,2})\b",
                )
                destination_order = _first_int(
                    plain,
                    r"\b(?:vi tri|thu tu|slot)\s+(\d{1,2})\b",
                )
                if resolved.destination_day is None and destination_day is not None:
                    updates["destination_day"] = destination_day
                if resolved.destination_order is None and destination_order is not None:
                    updates["destination_order"] = destination_order
                if resolved.target_day is None:
                    source_day = _source_day_for_move(
                        plain,
                        destination_day=destination_day,
                        has_named_target=parsed_name is not None,
                    )
                    if source_day is not None:
                        updates["target_day"] = source_day
            elif resolved.operation is PlanOperation.UPDATE_TIME:
                if resolved.start_time is None and slot_start is not None:
                    updates["start_time"] = slot_start
                if resolved.end_time is None and slot_end is not None:
                    updates["end_time"] = slot_end
            elif resolved.operation is PlanOperation.REPLACE_SLOT:
                if resolved.replacement_place_name is None:
                    updates["replacement_place_name"] = _requested_replacement_place(
                        plain
                    )
            elif resolved.operation is PlanOperation.SUGGEST_NEARBY:
                if resolved.anchor_place_name is None and parsed_name is not None:
                    updates["anchor_place_name"] = parsed_name
        return resolved.model_copy(update=updates)

    if active_plan is None:
        return PlanMutation(
            operation=(
                PlanOperation.UPDATE_CONSTRAINTS
                if budget is not None or occupancy
                else PlanOperation.NONE
            ),
            budget_vnd=budget,
            **occupancy,
        )

    day = _first_int(plain, r"\bngay\s+(\d{1,2})\b")
    order = _first_int(
        plain,
        r"\b(?:dia diem|diem|slot|muc)\s+(?:thu\s+)?(\d{1,2})\b",
    )
    role, period = _role_and_period(plain)
    matched_name = _mentioned_plan_place(plain, active_plan)
    common = {
        "expected_revision": active_plan.revision,
        "target_day": day,
        "target_order": order,
        "target_role": role if matched_name is None and order is None else None,
        "target_period": period if matched_name is None and order is None else None,
        "target_place_name": matched_name,
        "budget_vnd": budget,
        **occupancy,
    }

    if any(
        term in plain
        for term in (
            "xem lich trinh",
            "lich trinh hien tai",
            "lich trinh tung ngay",
            "lich trinh cho tung ngay",
            "chi tiet lich trinh",
            "plan hien tai",
        )
    ):
        return PlanMutation(operation=PlanOperation.QUERY_PLAN, **common)
    if any(
        term in plain
        for term in ("goi y gan", "goi y diem gan", "lan can", "xung quanh")
    ):
        return PlanMutation(
            operation=PlanOperation.SUGGEST_NEARBY,
            replacement_query=message,
            anchor_place_name=matched_name,
            **common,
        )
    slot_start_time, slot_end_time = _requested_slot_times(plain)
    if _is_move_command(plain):
        destination_day = _first_int(
            plain,
            r"\b(?:sang|qua|vao)\s+ngay\s+(\d{1,2})\b",
        )
        destination_order = _first_int(
            plain,
            r"\b(?:vi tri|thu tu|slot)\s+(\d{1,2})\b",
        )
        source_day = _source_day_for_move(
            plain,
            destination_day=destination_day,
            has_named_target=matched_name is not None,
        )
        return PlanMutation(
            operation=PlanOperation.MOVE_SLOT,
            destination_day=destination_day,
            destination_order=destination_order,
            **{
                **common,
                "target_day": source_day,
            },
        )
    if _is_add_command(plain):
        anchor_name = matched_name
        destination_day = (
            _first_int(
                plain,
                r"\b(?:vao|o|ngay)\s*(?:ngay\s+)?(\d{1,2})\b",
            )
            or day
        )
        destination_order = _first_int(
            plain,
            r"\b(?:vi tri|thu tu|slot)\s+(\d{1,2})\b",
        )
        return PlanMutation(
            operation=PlanOperation.ADD_SLOT,
            expected_revision=active_plan.revision,
            target_day=destination_day,
            target_order=destination_order,
            target_role=role,
            target_period=period,
            replacement_query=message,
            replacement_place_name=_requested_added_place(plain, anchor_name),
            anchor_place_name=anchor_name,
            start_time=slot_start_time,
            end_time=slot_end_time,
            budget_vnd=budget,
            **occupancy,
        )
    if _is_update_time_command(plain, slot_start_time, slot_end_time):
        return PlanMutation(
            operation=PlanOperation.UPDATE_TIME,
            start_time=slot_start_time,
            end_time=slot_end_time,
            **common,
        )
    start_time = _requested_time(plain, ("bat dau", "khoi hanh", "di tu"))
    end_time = _requested_time(plain, ("ket thuc", "ve luc", "dung luc"))
    if (
        (
            budget is not None
            or occupancy
            or start_time is not None
            or end_time is not None
        )
        and matched_name is None
        and order is None
    ):
        return PlanMutation(
            operation=PlanOperation.UPDATE_CONSTRAINTS,
            day_start_time=start_time,
            day_end_time=end_time,
            **common,
        )
    if any(term in plain for term in ("thay ", "doi ", "doi sang", "thay bang")):
        return PlanMutation(
            operation=PlanOperation.REPLACE_SLOT,
            replacement_query=message,
            replacement_place_name=_requested_replacement_place(plain),
            **common,
        )
    if any(term in plain for term in ("bo ", "xoa ", "khong di ")):
        return PlanMutation(operation=PlanOperation.REMOVE_SLOT, **common)
    if (
        budget is not None
        or occupancy
        or start_time is not None
        or end_time is not None
    ):
        return PlanMutation(
            operation=PlanOperation.UPDATE_CONSTRAINTS,
            day_start_time=start_time,
            day_end_time=end_time,
            **common,
        )
    if any(term in plain for term in ("lam lai ngay", "sap xep lai ngay")):
        return PlanMutation(operation=PlanOperation.REPLAN_DAY, **common)
    if any(term in plain for term in ("lam lai lich", "sap xep lai lich", "replan")):
        return PlanMutation(operation=PlanOperation.REPLAN_ALL, **common)
    return PlanMutation()


def build_active_trip_plan(
    *,
    itinerary: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    city: str,
    start_date: date | None,
    duration_days: int,
    operation: PlanOperation,
    previous: ActiveTripPlan | None = None,
    mutation: PlanMutation | None = None,
    travel_date_assumed: bool | None = None,
) -> ActiveTripPlan:
    now = datetime.now(timezone.utc)
    plan_id = previous.plan_id if previous is not None else f"trip_{uuid4().hex}"
    revision = previous.revision + 1 if previous is not None else 1
    constraints = (
        dict(previous.constraints)
        if previous is not None
        else {
            "party_size": 2,
            "rooms": 1,
            "currency": "VND",
        }
    )
    if mutation is not None:
        if mutation.day_start_time is not None:
            constraints["day_start_time"] = mutation.day_start_time
        if mutation.day_end_time is not None:
            constraints["day_end_time"] = mutation.day_end_time
        if mutation.budget_vnd is not None:
            constraints["budget_vnd"] = mutation.budget_vnd
        if mutation.adults is not None:
            constraints["adults"] = mutation.adults
            constraints["party_size"] = mutation.adults + (mutation.children or 0)
        if mutation.children is not None:
            constraints["children"] = mutation.children
            constraints["party_size"] = (
                mutation.adults or constraints.get("adults", 2)
            ) + mutation.children
        if mutation.rooms is not None:
            constraints["rooms"] = mutation.rooms
    if travel_date_assumed is not None:
        constraints["travel_date_assumed"] = travel_date_assumed

    days = _assign_stable_slot_ids(
        itinerary,
        plan_id=plan_id,
        previous=previous,
    )
    selected_places = _selected_place_snapshot(days, evidence, previous)
    days, budget_summary = annotate_itinerary_costs(
        days,
        selected_places,
        budget_vnd=_optional_int(constraints.get("budget_vnd")),
        party_size=_optional_int(constraints.get("party_size")) or 2,
        stay_nights=max(duration_days - 1, 0),
    )
    missing_price = budget_summary.status != "complete"
    has_unavailable_route = _has_missing_or_unavailable_route(days)
    status: Literal["complete", "partial", "draft"] = (
        "partial" if missing_price or has_unavailable_route else "complete"
    )
    return ActiveTripPlan(
        plan_id=plan_id,
        revision=revision,
        status=status,
        city=city,
        start_date=start_date,
        duration_days=duration_days,
        constraints=constraints,
        itinerary=days,
        selected_places=selected_places,
        budget_summary=budget_summary,
        last_operation=operation,
        created_at=previous.created_at if previous is not None else now,
        updated_at=now,
    )


def apply_plan_mutation(
    plan: ActiveTripPlan,
    mutation: PlanMutation,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PlanChange]:
    days = deepcopy(plan.itinerary)
    selected_places = deepcopy(plan.selected_places)
    target = locate_target_slot(days, mutation)
    previous_ids = _slot_ids(days)

    if mutation.operation is PlanOperation.UPDATE_CONSTRAINTS:
        return (
            days,
            selected_places,
            PlanChange(
                operation=mutation.operation,
                previous_revision=plan.revision,
                preserved_slot_ids=previous_ids,
                message="updated_constraints",
            ),
        )
    if mutation.operation is PlanOperation.ADD_SLOT:
        candidate = choose_add_candidate(candidates, days=days, mutation=mutation)
        if candidate is None:
            raise ValueError("plan_add_candidate_not_found")
        destination = _locate_destination_day(days, mutation.target_day)
        if destination is None:
            raise ValueError("plan_destination_day_not_found")
        slots = destination.setdefault("slots", [])
        insert_index = _insertion_index(mutation.target_order, len(slots))
        if mutation.anchor_place_name:
            anchor = _locate_named_slot(days, mutation.anchor_place_name)
            if anchor is not None:
                anchor_day, anchor_index, _ = anchor
                if mutation.target_day is None or anchor_day is destination:
                    destination = anchor_day
                    slots = destination.setdefault("slots", [])
                    insert_index = anchor_index + 1
        added_slot = _slot_from_candidate(
            candidate,
            plan_id=plan.plan_id,
            mutation=mutation,
            fallback_start_time=_insertion_start_time(slots, insert_index),
        )
        slots.insert(insert_index, added_slot)
        _renumber(destination)
        selected_places.append(_compact_place(candidate))
        added_id = str(added_slot["slot_id"])
        return (
            days,
            selected_places,
            PlanChange(
                operation=mutation.operation,
                previous_revision=plan.revision,
                target_slot_id=added_id,
                changed_slot_ids=[added_id],
                preserved_slot_ids=previous_ids,
                message=f"added:{candidate['place_id']}",
            ),
        )
    if target is None:
        raise ValueError("plan_target_slot_not_found")
    day_item, slot_index, slot = target

    if mutation.operation is PlanOperation.REMOVE_SLOT:
        removed_id = str(slot.get("slot_id") or "")
        previous_id = (
            str(day_item["slots"][slot_index - 1].get("slot_id") or "")
            if slot_index > 0
            else ""
        )
        day_item["slots"].pop(slot_index)
        _renumber(day_item)
        retained_ids = {
            str(item.get("place_id")) for day in days for item in day.get("slots", [])
        }
        selected_places = [
            item
            for item in selected_places
            if str(item.get("place_id")) in retained_ids
        ]
        return (
            days,
            selected_places,
            PlanChange(
                operation=mutation.operation,
                previous_revision=plan.revision,
                target_slot_id=removed_id,
                changed_slot_ids=[item for item in (removed_id, previous_id) if item],
                preserved_slot_ids=[
                    item for item in previous_ids if item != removed_id
                ],
                message="removed_slot",
            ),
        )

    if mutation.operation is PlanOperation.MOVE_SLOT:
        moved_id = str(slot.get("slot_id") or "")
        old_previous_id = (
            str(day_item["slots"][slot_index - 1].get("slot_id") or "")
            if slot_index > 0
            else ""
        )
        day_item["slots"].pop(slot_index)
        _renumber(day_item)
        destination = _locate_destination_day(
            days,
            mutation.destination_day
            if mutation.destination_day is not None
            else int(day_item.get("day") or 0),
        )
        if destination is None:
            raise ValueError("plan_destination_day_not_found")
        destination_slots = destination.setdefault("slots", [])
        destination_index = _insertion_index(
            mutation.destination_order,
            len(destination_slots),
        )
        destination_slots.insert(destination_index, slot)
        _renumber(destination)
        changed = list(
            dict.fromkeys(item for item in (moved_id, old_previous_id) if item)
        )
        return (
            days,
            selected_places,
            PlanChange(
                operation=mutation.operation,
                previous_revision=plan.revision,
                target_slot_id=moved_id or None,
                changed_slot_ids=changed,
                preserved_slot_ids=[
                    item for item in previous_ids if item not in set(changed)
                ],
                message=(
                    f"moved:{moved_id}:d{int(destination.get('day') or 0)}:"
                    f"s{destination_index + 1}"
                ),
            ),
        )

    if mutation.operation is PlanOperation.UPDATE_TIME:
        if mutation.start_time is None and mutation.end_time is None:
            raise ValueError("plan_time_not_provided")
        start_time, end_time = _updated_slot_times(slot, mutation)
        slot["start_time"] = start_time
        slot["end_time"] = end_time
        slot_id = str(slot.get("slot_id") or "")
        return (
            days,
            selected_places,
            PlanChange(
                operation=mutation.operation,
                previous_revision=plan.revision,
                target_slot_id=slot_id or None,
                changed_slot_ids=[slot_id] if slot_id else [],
                preserved_slot_ids=[item for item in previous_ids if item != slot_id],
                message=f"updated_time:{slot_id}:{start_time}:{end_time}",
            ),
        )

    if mutation.operation is not PlanOperation.REPLACE_SLOT:
        raise ValueError(f"unsupported_plan_mutation:{mutation.operation.value}")
    replacement = choose_replacement_candidate(
        candidates,
        days=days,
        target_slot=slot,
        mutation=mutation,
    )
    if replacement is None:
        raise ValueError("plan_replacement_candidate_not_found")
    replacement_id = str(replacement.get("place_id") or "")
    old_id = str(slot.get("place_id") or "")
    updated_slot = {
        **slot,
        "place_id": replacement_id,
        "name": replacement.get("name") or slot.get("name"),
        "city": replacement.get("city") or slot.get("city"),
        "entity_type": replacement.get("entity_type") or slot.get("entity_type"),
        "rationale": "Địa điểm được thay theo yêu cầu; các slot khác được giữ nguyên.",
        "transport_to_next": None,
        "hotel_availability": (
            (replacement.get("attributes") or {}).get("hotel_availability")
            if replacement.get("entity_type") == "hotel"
            else None
        ),
        "cost_estimate": None,
    }
    day_item["slots"][slot_index] = updated_slot
    selected_places = [
        item for item in selected_places if str(item.get("place_id")) != old_id
    ]
    selected_places.append(_compact_place(replacement))
    slot_id = str(slot.get("slot_id") or "")
    return (
        days,
        selected_places,
        PlanChange(
            operation=mutation.operation,
            previous_revision=plan.revision,
            target_slot_id=slot_id or None,
            changed_slot_ids=[slot_id] if slot_id else [],
            preserved_slot_ids=[item for item in previous_ids if item != slot_id],
            message=f"replaced:{old_id}:{replacement_id}",
        ),
    )


def locate_target_slot(
    itinerary: list[dict[str, Any]],
    mutation: PlanMutation,
) -> tuple[dict[str, Any], int, dict[str, Any]] | None:
    candidates: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    target_name = normalize_text(mutation.target_place_name or "")
    for day_item in itinerary:
        if (
            mutation.target_day is not None
            and int(day_item.get("day") or 0) != mutation.target_day
        ):
            continue
        for index, slot in enumerate(day_item.get("slots", [])):
            if (
                mutation.target_slot_id
                and slot.get("slot_id") != mutation.target_slot_id
            ):
                continue
            if (
                mutation.target_order is not None
                and int(slot.get("order") or 0) != mutation.target_order
            ):
                continue
            if (
                mutation.target_role
                and str(slot.get("role") or "") != mutation.target_role
            ):
                continue
            if target_name and target_name not in normalize_text(
                str(slot.get("name") or "")
            ):
                continue
            if mutation.target_period and not _slot_matches_period(
                slot, mutation.target_period
            ):
                continue
            candidates.append((day_item, index, slot))
    return candidates[0] if len(candidates) == 1 else None


def choose_replacement_candidate(
    candidates: list[dict[str, Any]],
    *,
    days: list[dict[str, Any]],
    target_slot: Mapping[str, Any],
    mutation: PlanMutation,
) -> dict[str, Any] | None:
    used_ids = {
        str(slot.get("place_id") or "")
        for day_item in days
        for slot in day_item.get("slots", [])
    }
    target_type = mutation.entity_type or str(target_slot.get("entity_type") or "")
    desired = normalize_text(mutation.replacement_place_name or "")
    eligible = [
        item
        for item in candidates
        if item.get("place_id")
        and str(item.get("place_id")) not in used_ids
        and (not target_type or item.get("entity_type") == target_type)
    ]
    if desired:
        exact = [
            item
            for item in eligible
            if desired in normalize_text(str(item.get("name") or ""))
            or normalize_text(str(item.get("name") or "")) in desired
        ]
        if exact:
            eligible = exact
    return min(
        eligible,
        key=lambda item: (
            _optional_float(item.get("distance_km")) is None,
            _optional_float(item.get("distance_km")) or math.inf,
            -(_optional_float(item.get("score")) or 0),
            str(item.get("place_id")),
        ),
        default=None,
    )


def choose_add_candidate(
    candidates: list[dict[str, Any]],
    *,
    days: list[dict[str, Any]],
    mutation: PlanMutation,
) -> dict[str, Any] | None:
    """Choose an add target exclusively from provider-grounded candidates."""

    used_ids = {
        str(slot.get("place_id") or "")
        for day_item in days
        for slot in day_item.get("slots", [])
    }
    desired = normalize_text(mutation.replacement_place_name or "")
    eligible = [
        item
        for item in candidates
        if item.get("place_id")
        and str(item.get("place_id")) not in used_ids
        and (
            not mutation.entity_type
            or str(item.get("entity_type") or "") == mutation.entity_type
        )
    ]
    if desired:
        eligible = [
            item
            for item in eligible
            if desired in normalize_text(str(item.get("name") or ""))
            or normalize_text(str(item.get("name") or "")) in desired
        ]
    return min(
        eligible,
        key=lambda item: (
            _optional_float(item.get("distance_km")) is None,
            _optional_float(item.get("distance_km")) or math.inf,
            -(_optional_float(item.get("score")) or 0),
            str(item.get("place_id")),
        ),
        default=None,
    )


def _slot_from_candidate(
    candidate: Mapping[str, Any],
    *,
    plan_id: str,
    mutation: PlanMutation,
    fallback_start_time: str,
) -> dict[str, Any]:
    entity_type = str(candidate.get("entity_type") or mutation.entity_type or "")
    role = (
        mutation.target_role
        or str(candidate.get("role") or "")
        or {
            "attraction": "activity",
            "restaurant": "meal",
            "cafe": "cafe_break",
            "nightlife": "nightlife",
            "hotel": "check_in",
        }.get(entity_type)
    )
    attributes = candidate.get("attributes")
    attrs = attributes if isinstance(attributes, Mapping) else {}
    duration = {
        "activity": 90,
        "meal": 75,
        "cafe_break": 60,
        "nightlife": 90,
        "check_in": 30,
    }.get(str(role or ""), 60)
    start_minutes = _time_minutes(mutation.start_time)
    end_minutes = _time_minutes(mutation.end_time)
    if start_minutes is None and end_minutes is not None:
        start_minutes = max(0, end_minutes - duration)
    if start_minutes is None:
        start_minutes = _time_minutes(fallback_start_time) or 9 * 60
    if end_minutes is None:
        end_minutes = start_minutes + duration
    if start_minutes >= end_minutes or end_minutes >= 24 * 60:
        raise ValueError("plan_add_time_range_invalid")
    return {
        "slot_id": f"{plan_id}:slot:{uuid4().hex}",
        "order": 0,
        "start_time": _format_time(start_minutes),
        "end_time": _format_time(end_minutes),
        "place_id": str(candidate["place_id"]),
        "name": candidate.get("name"),
        "city": candidate.get("city"),
        "entity_type": entity_type or None,
        "role": role,
        "rationale": "Địa điểm được thêm theo yêu cầu; các slot khác được giữ nguyên.",
        "transport_to_next": None,
        "hotel_availability": (
            attrs.get("hotel_availability") if entity_type == "hotel" else None
        ),
        "cost_estimate": None,
    }


def _locate_destination_day(
    days: list[dict[str, Any]],
    day_number: int | None,
) -> dict[str, Any] | None:
    if day_number is None:
        return days[0] if len(days) == 1 else None
    return next(
        (item for item in days if int(item.get("day") or 0) == day_number),
        None,
    )


def _locate_named_slot(
    days: list[dict[str, Any]],
    place_name: str,
) -> tuple[dict[str, Any], int, dict[str, Any]] | None:
    desired = normalize_text(place_name)
    matches = [
        (day_item, index, slot)
        for day_item in days
        for index, slot in enumerate(day_item.get("slots", []))
        if desired
        and (
            desired in normalize_text(str(slot.get("name") or ""))
            or normalize_text(str(slot.get("name") or "")) in desired
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _insertion_index(order: int | None, size: int) -> int:
    if order is None:
        return size
    return max(0, min(order - 1, size))


def _insertion_start_time(slots: list[dict[str, Any]], index: int) -> str:
    if index > 0:
        previous_end = _time_minutes(slots[index - 1].get("end_time"))
        if previous_end is not None:
            return _format_time(min(previous_end + 15, 23 * 60)) or "09:00"
    if index < len(slots):
        next_start = _time_minutes(slots[index].get("start_time"))
        if next_start is not None:
            return _format_time(max(0, next_start - 75)) or "09:00"
    return "09:00"


def _updated_slot_times(
    slot: Mapping[str, Any],
    mutation: PlanMutation,
) -> tuple[str | None, str | None]:
    old_start = _time_minutes(slot.get("start_time"))
    old_end = _time_minutes(slot.get("end_time"))
    new_start = _time_minutes(mutation.start_time)
    new_end = _time_minutes(mutation.end_time)
    if new_start is not None and new_end is None and old_start is not None and old_end:
        duration = old_end - old_start
        if duration > 0 and new_start + duration < 24 * 60:
            new_end = new_start + duration
    if new_start is None:
        new_start = old_start
    if new_end is None:
        new_end = old_end
    if new_start is not None and new_end is not None and new_start >= new_end:
        raise ValueError("plan_time_range_invalid")
    return _format_time(new_start), _format_time(new_end)


def annotate_itinerary_costs(
    itinerary: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    budget_vnd: int | None,
    party_size: int,
    stay_nights: int,
) -> tuple[list[dict[str, Any]], BudgetSummary]:
    days = deepcopy(itinerary)
    by_id = {
        str(item.get("place_id")): item for item in evidence if item.get("place_id")
    }
    min_total = 0
    max_total = 0
    priced = 0
    estimable = 0
    missing: list[str] = []
    counted_hotels: set[str] = set()
    mandatory_hotel_present = False
    mandatory_hotel_priced = False
    transport_fares_unpriced = any(
        len(day_item.get("slots", [])) > 1 for day_item in days
    )
    for day_item in days:
        for slot in day_item.get("slots", []):
            place_id = str(slot.get("place_id") or "")
            place = by_id.get(place_id, {})
            is_stay_hotel = (
                slot.get("entity_type") == "hotel" and slot.get("role") != "check_out"
            )
            if is_stay_hotel:
                mandatory_hotel_present = True
            estimate = _cost_estimate(
                slot,
                place,
                party_size=party_size,
                stay_nights=stay_nights,
            )
            slot["cost_estimate"] = estimate
            if slot.get("role") == "check_out":
                estimate["included_in_budget"] = False
                continue
            if slot.get("entity_type") == "hotel":
                if place_id in counted_hotels:
                    estimate["included_in_budget"] = False
                    continue
                counted_hotels.add(place_id)
            estimable += 1
            minimum = _optional_int(estimate.get("amount_min"))
            maximum = _optional_int(estimate.get("amount_max"))
            if minimum is None or maximum is None:
                if place_id and place_id not in missing:
                    missing.append(place_id)
                continue
            priced += 1
            min_total += minimum
            max_total += maximum
            if is_stay_hotel:
                mandatory_hotel_priced = True

    mandatory_stay_unpriced = stay_nights > 0 and not mandatory_hotel_priced
    if stay_nights > 0 and not mandatory_hotel_present:
        estimable += 1
    if transport_fares_unpriced:
        estimable += 1

    if priced == 0:
        status: Literal["complete", "partial", "unavailable"] = "unavailable"
        estimated_min = None
        estimated_max = None
    else:
        status = (
            "complete"
            if not missing
            and not mandatory_stay_unpriced
            and not transport_fares_unpriced
            else "partial"
        )
        estimated_min = min_total
        estimated_max = max_total
    within_budget = None
    remaining_min = None
    remaining_max = None
    if (
        budget_vnd is not None
        and estimated_min is not None
        and estimated_max is not None
        and not mandatory_stay_unpriced
        and not transport_fares_unpriced
    ):
        if estimated_min > budget_vnd:
            within_budget = False
        elif status == "complete" and estimated_max <= budget_vnd:
            within_budget = True
        remaining_min = budget_vnd - estimated_max
        remaining_max = budget_vnd - estimated_min
    exclusions: list[str] = []
    if transport_fares_unpriced:
        exclusions.append("transport_fares_not_available")
    if missing:
        exclusions.append("places_without_numeric_price")
    if mandatory_stay_unpriced:
        exclusions.append("mandatory_hotel_stay_not_priced")
    return days, BudgetSummary(
        status=status,
        party_size=party_size,
        budget_amount=budget_vnd,
        estimated_min=estimated_min,
        estimated_max=estimated_max,
        within_budget=within_budget,
        remaining_min=remaining_min,
        remaining_max=remaining_max,
        priced_item_count=priced,
        estimable_item_count=estimable,
        missing_place_ids=missing,
        exclusions=exclusions,
    )


def extract_budget_vnd(message: str) -> int | None:
    plain = normalize_text(message).replace(",", ".")
    cue = r"(?:budget|ngan sach|chi phi|toi da|khong qua|tam|khoang)"
    match = re.search(
        rf"{cue}[^\d]{{0,20}}(\d+(?:\.\d+)?)\s*(trieu|tr|k|nghin|ngan|vnd|d|dong)?\b",
        plain,
    )
    if match is None:
        match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(trieu|tr|k|nghin|ngan|vnd|dong)\b", plain
        )
    if match is None:
        return None
    value = Decimal(match.group(1))
    unit = match.group(2) or "vnd"
    multiplier = Decimal(1)
    if unit in {"trieu", "tr"}:
        multiplier = Decimal(1_000_000)
    elif unit in {"k", "nghin", "ngan"}:
        multiplier = Decimal(1_000)
    return int(value * multiplier)


def extract_occupancy(message: str) -> dict[str, int]:
    plain = normalize_text(message)
    result: dict[str, int] = {}
    adult = re.search(r"\b(\d{1,2})\s*(?:nguoi lon|adult)", plain)
    child = re.search(r"\b(\d{1,2})\s*(?:tre em|tre nho|child)", plain)
    people = re.search(r"\b(\d{1,2})\s*(?:nguoi|khach)\b", plain)
    rooms = re.search(r"\b(\d{1,2})\s*(?:phong|room)\b", plain)
    if adult:
        result["adults"] = int(adult.group(1))
    elif people:
        result["adults"] = int(people.group(1))
    if child:
        result["children"] = int(child.group(1))
    if rooms:
        result["rooms"] = int(rooms.group(1))
    return result


def normalize_transport_units(itinerary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days = deepcopy(itinerary)
    for day_item in days:
        for slot in day_item.get("slots", []):
            value = slot.get("transport_to_next")
            if not isinstance(value, dict):
                continue
            distance = _optional_float(value.get("distance_meters"))
            duration = _optional_float(value.get("duration_seconds"))
            if distance is not None:
                value["distance_km"] = round(distance / 1000, 1)
            if duration is not None:
                value["duration_minutes"] = max(1, math.ceil(duration / 60))
            mode = str(value.get("recommended_mode") or "")
            value["mode_label"] = {
                "walk": "Đi bộ",
                "bicycle": "Xe đạp",
                "two_wheeler": "Xe máy",
                "drive": "Ô tô/taxi",
            }.get(mode, mode or None)
    return days


def _assign_stable_slot_ids(
    itinerary: list[dict[str, Any]],
    *,
    plan_id: str,
    previous: ActiveTripPlan | None,
) -> list[dict[str, Any]]:
    days = deepcopy(itinerary)
    previous_by_position: dict[tuple[int, int], str] = {}
    if previous is not None:
        for day_item in previous.itinerary:
            day = int(day_item.get("day") or 0)
            for slot in day_item.get("slots", []):
                slot_id = str(slot.get("slot_id") or "")
                if slot_id:
                    previous_by_position[(day, int(slot.get("order") or 0))] = slot_id
    for day_item in days:
        day = int(day_item.get("day") or 1)
        for index, slot in enumerate(day_item.get("slots", []), start=1):
            slot["order"] = index
            existing = str(slot.get("slot_id") or "")
            slot["slot_id"] = (
                existing
                or previous_by_position.get((day, index))
                or f"{plan_id}:d{day}:s{index}"
            )
    return normalize_transport_units(days)


def _selected_place_snapshot(
    itinerary: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    previous: ActiveTripPlan | None,
) -> list[dict[str, Any]]:
    ids = {
        str(slot.get("place_id") or "")
        for day_item in itinerary
        for slot in day_item.get("slots", [])
    }
    by_id: dict[str, dict[str, Any]] = {}
    if previous is not None:
        by_id.update(
            {
                str(item.get("place_id")): dict(item)
                for item in previous.selected_places
                if item.get("place_id")
            }
        )
    for item in evidence:
        place_id = str(item.get("place_id") or "")
        if not place_id:
            continue
        incoming = _compact_place(item)
        existing = by_id.get(place_id)
        by_id[place_id] = (
            _merge_place_snapshot(existing, incoming)
            if existing is not None
            else incoming
        )
    return [by_id[place_id] for place_id in sorted(ids) if place_id in by_id]


def _compact_place(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(item.get(key))
        for key in (
            "place_id",
            "name",
            "city",
            "entity_type",
            "category",
            "score",
            "distance_km",
            "source",
            "attributes",
        )
        if item.get(key) is not None
    }


def _merge_place_snapshot(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge fresh evidence without discarding richer persisted attributes."""

    merged = deepcopy(dict(existing))
    for key, value in incoming.items():
        previous = merged.get(key)
        if key == "attributes" and isinstance(value, Mapping):
            previous_attributes = previous if isinstance(previous, Mapping) else {}
            merged[key] = _merge_place_attributes(previous_attributes, value)
        elif isinstance(value, Mapping) and isinstance(previous, Mapping):
            merged[key] = _deep_merge_mapping(previous, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _merge_place_attributes(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(dict(existing))
    for key, value in incoming.items():
        # These envelopes represent one operational context and must be
        # replaced atomically when a fresher context is available.
        if key in {"current", "hotel_availability"}:
            merged[key] = deepcopy(value)
            continue
        previous = merged.get(key)
        if isinstance(value, Mapping) and isinstance(previous, Mapping):
            merged[key] = _deep_merge_mapping(previous, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _deep_merge_mapping(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(dict(existing))
    for key, value in incoming.items():
        previous = merged.get(key)
        if isinstance(value, Mapping) and isinstance(previous, Mapping):
            merged[key] = _deep_merge_mapping(previous, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _cost_estimate(
    slot: Mapping[str, Any],
    place: Mapping[str, Any],
    *,
    party_size: int,
    stay_nights: int,
) -> dict[str, Any]:
    entity_type = str(slot.get("entity_type") or place.get("entity_type") or "")
    attributes = place.get("attributes")
    attrs = dict(attributes) if isinstance(attributes, Mapping) else {}
    if entity_type == "hotel":
        availability = slot.get("hotel_availability") or attrs.get("hotel_availability")
        estimate = _hotel_cost(availability, stay_nights=stay_nights)
        if estimate is not None:
            return estimate

    numeric = _numeric_place_price(attrs, entity_type=entity_type)
    if numeric is not None:
        minimum, maximum, basis, source = numeric
        if basis == "per_person":
            minimum *= party_size
            maximum *= party_size
        return {
            "status": "exact" if minimum == maximum else "range",
            "currency": "VND",
            "amount_min": minimum,
            "amount_max": maximum,
            "basis": basis,
            "party_size": party_size if basis == "per_person" else None,
            "source": source,
            "included_in_budget": True,
        }
    tier = _price_level(attrs)
    return {
        "status": "tier" if tier is not None else "unknown",
        "currency": "VND",
        "amount_min": None,
        "amount_max": None,
        "basis": "price_level" if tier is not None else "unknown",
        "price_level": tier,
        "source": "google_maps" if tier is not None else None,
        "included_in_budget": False,
    }


def _hotel_cost(value: object, *, stay_nights: int) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    windows = value.get("windows")
    selected_index = value.get("selected_window_index")
    if not isinstance(windows, list) or not isinstance(selected_index, int):
        return None
    if selected_index < 0 or selected_index >= len(windows):
        return None
    window = windows[selected_index]
    if not isinstance(window, Mapping):
        return None
    offers = [item for item in window.get("offers", []) if isinstance(item, Mapping)]
    priced: list[tuple[int, int, Mapping[str, Any], str]] = []
    for offer in offers:
        total = _money(offer.get("total_amount"))
        nightly = _money(offer.get("nightly_amount")) or _money(offer.get("amount"))
        minimum = _money(offer.get("min_amount"))
        maximum = _money(offer.get("max_amount"))
        if total is not None:
            priced.append((total, total, offer, "per_stay"))
        elif nightly is not None:
            value_total = nightly * max(stay_nights, 1)
            priced.append(
                (value_total, value_total, offer, "per_stay_derived_from_nightly")
            )
        elif minimum is not None and maximum is not None:
            priced.append((minimum, maximum, offer, "per_stay_range"))
    if not priced:
        return None
    minimum, maximum, offer, basis = min(priced, key=lambda item: (item[1], item[0]))
    return {
        "status": "exact" if minimum == maximum else "range",
        "currency": str(offer.get("currency") or "VND"),
        "amount_min": minimum,
        "amount_max": maximum,
        "basis": basis,
        "nightly_amount": _money(offer.get("nightly_amount") or offer.get("amount")),
        "stay_nights": stay_nights,
        "requested_check_in": window.get("requested_check_in"),
        "check_in": window.get("check_in"),
        "check_out": window.get("check_out"),
        "fallback_offset_days": window.get("fallback_offset_days"),
        "seller": offer.get("seller"),
        "observed_at": offer.get("observed_at"),
        "stale_after": offer.get("stale_after"),
        "source": "trivago_current",
        "included_in_budget": True,
    }


def _numeric_place_price(
    attributes: Mapping[str, Any],
    *,
    entity_type: str,
) -> tuple[int, int, str, str] | None:
    current = attributes.get("current")
    current_map = current if isinstance(current, Mapping) else {}
    sources = [current_map, attributes]
    exact_keys = (
        "ticket_price_adult",
        "admission_fee",
        "entry_fee",
        "price_per_person",
        "average_price",
        "drink_price",
    )
    for source in sources:
        for key in exact_keys:
            amount = _money(source.get(key))
            if amount is not None:
                basis = "per_person" if entity_type != "hotel" else "per_stay"
                return (
                    amount,
                    amount,
                    basis,
                    "current_place" if source is current_map else "knowledge_base",
                )
        for minimum_key, maximum_key in (
            ("price_min", "price_max"),
            ("price_range_min", "price_range_max"),
            ("price_per_person_min", "price_per_person_max"),
            ("drink_price_min", "drink_price_max"),
            ("ticket_price_min", "ticket_price_max"),
            ("entry_fee_min", "entry_fee_max"),
        ):
            minimum = _money(source.get(minimum_key))
            maximum = _money(source.get(maximum_key))
            if minimum is not None and maximum is not None:
                return (
                    minimum,
                    maximum,
                    "per_person",
                    "current_place" if source is current_map else "knowledge_base",
                )
        parsed = _money_range(source.get("price_range"))
        if parsed is not None:
            return (
                parsed[0],
                parsed[1],
                "per_person",
                "current_place" if source is current_map else "knowledge_base",
            )
    google_price = attributes.get("google_maps_price")
    if isinstance(google_price, Mapping):
        parsed = _money_range(google_price.get("raw_text"))
        if parsed is not None:
            return parsed[0], parsed[1], "per_person", "google_maps"
    return None


def _price_level(attributes: Mapping[str, Any]) -> int | None:
    current = attributes.get("current")
    values = [
        current.get("price_level") if isinstance(current, Mapping) else None,
        attributes.get("price_level"),
    ]
    google_price = attributes.get("google_maps_price")
    if isinstance(google_price, Mapping):
        values.append(google_price.get("level"))
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None and 1 <= parsed <= 4:
            return parsed
    return None


def _money_range(value: object) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        minimum = _money(value.get("min", value.get("minimum")))
        maximum = _money(value.get("max", value.get("maximum")))
        if minimum is not None and maximum is not None:
            return minimum, maximum
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        minimum, maximum = _money(value[0]), _money(value[1])
        if minimum is not None and maximum is not None:
            return minimum, maximum
    if not isinstance(value, str):
        return None
    plain = normalize_text(value).replace(",", ".")
    numbers = re.findall(r"\d+(?:\.\d+)?\s*(?:trieu|tr|k|nghin|ngan)?", plain)
    parsed = [_money(item) for item in numbers]
    amounts = [item for item in parsed if item is not None]
    if len(amounts) >= 2:
        return min(amounts[0], amounts[1]), max(amounts[0], amounts[1])
    return None


def _money(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return max(0, int(value))
    if not isinstance(value, str):
        return None
    plain = normalize_text(value).strip().replace("₫", "").replace("đ", "")
    if plain in {"free", "mien phi", "0"}:
        return 0
    unit_match = re.search(r"(trieu|tr|k|nghin|ngan)\b", plain)
    unit = unit_match.group(1) if unit_match else None
    cleaned = re.sub(r"[^0-9.,]", "", plain)
    if not cleaned:
        return None
    try:
        if unit:
            number = Decimal(cleaned.replace(",", "."))
        else:
            # Vietnamese separators in provider text are thousands separators.
            normalized = cleaned.replace(".", "").replace(",", "")
            number = Decimal(normalized)
    except InvalidOperation:
        return None
    multiplier = (
        Decimal(1_000_000)
        if unit in {"trieu", "tr"}
        else Decimal(1_000)
        if unit
        else Decimal(1)
    )
    return max(0, int(number * multiplier))


def _mentioned_plan_place(plain: str, plan: ActiveTripPlan) -> str | None:
    matches = [
        str(slot.get("name"))
        for day_item in plan.itinerary
        for slot in day_item.get("slots", [])
        if slot.get("name") and normalize_text(str(slot.get("name"))) in plain
    ]
    return max(matches, key=len, default=None)


def _role_and_period(plain: str) -> tuple[str | None, Any]:
    if any(term in plain for term in ("bua toi", "an toi", "buoi toi")):
        return "meal", "dinner"
    if any(term in plain for term in ("bua trua", "an trua")):
        return "meal", "lunch"
    if any(term in plain for term in ("ca phe", "cafe", "uong tra")):
        return "cafe_break", "afternoon"
    if any(term in plain for term in ("tham quan", "di choi", "dia diem")):
        return "activity", None
    if "khach san" in plain:
        return "check_in", None
    return None, None


def _is_add_command(plain: str) -> bool:
    if not re.search(r"\b(?:them|chen|bo sung)\b", plain):
        return False
    has_plan_position = any(
        term in plain
        for term in (
            "vao lich",
            "lich trinh",
            "ngay",
            "slot",
            "vi tri",
            "sau ",
            "truoc ",
            "luc ",
        )
    )
    if re.search(r"\b(?:goi y|de xuat)\b", plain) and not has_plan_position:
        return False
    has_generic_plan_item = bool(
        re.search(
            r"\b(?:dia diem|diem den|diem dung|hoat dong)"
            r"(?:\s+(?:moi|khac|nua))?\b",
            plain,
        )
    )
    return has_plan_position or has_generic_plan_item


def _is_move_command(plain: str) -> bool:
    if not re.search(r"\b(?:chuyen|doi vi tri|dua)\b", plain):
        return False
    return any(
        term in plain
        for term in (
            "sang ngay",
            "qua ngay",
            "vao ngay",
            "vi tri",
            "thu tu",
            "len truoc",
            "xuong sau",
        )
    )


def _is_update_time_command(
    plain: str,
    start_time: str | None,
    end_time: str | None,
) -> bool:
    if start_time is None and end_time is None:
        return False
    return any(
        term in plain
        for term in (
            "doi gio",
            "doi thoi gian",
            "chuyen gio",
            "doi luc",
            "sang luc",
            "vao luc",
            "bat dau luc",
            "ket thuc luc",
        )
    )


def _requested_slot_times(plain: str) -> tuple[str | None, str | None]:
    matches = list(
        re.finditer(
            r"(?<!\d)([01]?\d|2[0-3])"
            r"(?::([0-5]\d)|h(?:([0-5]\d))?|\s*gio(?:\s*([0-5]?\d))?)\b",
            plain,
        )
    )
    times = [
        f"{int(match.group(1)):02d}:"
        f"{int(match.group(2) or match.group(3) or match.group(4) or 0):02d}"
        for match in matches
    ]
    if not times:
        return None, None
    if len(times) >= 2:
        return times[0], times[1]
    only = times[0]
    if any(term in plain for term in ("gio ket thuc", "ket thuc luc", "den luc")):
        return None, only
    return only, None


def _requested_added_place(plain: str, anchor_name: str | None) -> str | None:
    match = re.search(
        r"\b(?:them|chen|bo sung)\s+(.+?)"
        r"(?=\s+(?:vao|o|ngay|luc|sau|truoc|vi tri|slot)\b|$)",
        plain,
    )
    if match is None:
        return None
    requested = match.group(1).strip(" ,.-")
    if anchor_name and normalize_text(anchor_name) == requested:
        return None
    if re.fullmatch(
        r"(?:(?:mot|cac|nhung)\s+)?"
        r"(?:dia diem|diem den|diem dung|hoat dong)"
        r"(?:\s+(?:moi|khac|nua))*",
        requested,
    ):
        return None
    return requested or None


def _requested_replacement_place(plain: str) -> str | None:
    match = re.search(
        r"\b(?:bang|sang|thanh)\s+(.+?)"
        r"(?=\s+(?:vao|o|ngay|luc|vi tri|slot)\b|$)",
        plain,
    )
    if match is None:
        return None
    requested = match.group(1).strip(" ,.-")
    return requested or None


def _source_day_for_move(
    plain: str,
    *,
    destination_day: int | None,
    has_named_target: bool,
) -> int | None:
    explicit = _first_int(
        plain,
        r"\b(?:tu|o)\s+ngay\s+(\d{1,2})\b",
    )
    if explicit is not None:
        return explicit
    days = [int(value) for value in re.findall(r"\bngay\s+(\d{1,2})\b", plain)]
    if len(days) >= 2:
        return days[0]
    if has_named_target:
        return None
    if len(days) == 1 and days[0] != destination_day:
        return days[0]
    return None


def _requested_time(plain: str, cues: tuple[str, ...]) -> str | None:
    for cue in cues:
        match = re.search(
            rf"{re.escape(cue)}[^\d]{{0,12}}(\d{{1,2}})(?::(\d{{2}}))?\s*(?:gio|h)?",
            plain,
        )
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
    return None


def _slot_matches_period(slot: Mapping[str, Any], period: str) -> bool:
    value = str(slot.get("start_time") or "00:00")
    try:
        hour = int(value.split(":", 1)[0])
    except (ValueError, IndexError):
        return False
    return {
        "morning": hour < 11,
        "lunch": 10 <= hour < 15,
        "afternoon": 12 <= hour < 18,
        "dinner": 16 <= hour < 21,
        "evening": hour >= 18,
    }.get(period, False)


def _first_int(value: str, pattern: str) -> int | None:
    match = re.search(pattern, value)
    return int(match.group(1)) if match else None


def _time_minutes(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", value)
    if match is None:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _format_time(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value // 60:02d}:{value % 60:02d}"


def _renumber(day_item: dict[str, Any]) -> None:
    for index, slot in enumerate(day_item.get("slots", []), start=1):
        slot["order"] = index


def _slot_ids(days: list[dict[str, Any]]) -> list[str]:
    return [
        str(slot.get("slot_id"))
        for day_item in days
        for slot in day_item.get("slots", [])
        if slot.get("slot_id")
    ]


def invalidate_adjacent_routes(
    itinerary: list[dict[str, Any]],
    changed_slot_ids: list[str],
) -> list[dict[str, Any]]:
    """Invalidate only legs whose origin or destination changed."""

    days = deepcopy(itinerary)
    changed = set(changed_slot_ids)
    for day_item in days:
        slots = day_item.get("slots", [])
        for index, slot in enumerate(slots):
            if str(slot.get("slot_id") or "") not in changed:
                continue
            slot["transport_to_next"] = None
            if index > 0:
                slots[index - 1]["transport_to_next"] = None
    return days


def _has_missing_or_unavailable_route(itinerary: list[dict[str, Any]]) -> bool:
    for day_item in itinerary:
        slots = day_item.get("slots", [])
        for origin, destination in zip(slots, slots[1:]):
            if not origin.get("place_id") or not destination.get("place_id"):
                continue
            if origin.get("place_id") == destination.get("place_id"):
                continue
            route = origin.get("transport_to_next")
            if not isinstance(route, Mapping):
                return True
            if route.get("status") not in {"recommended", "available"}:
                return True
            if route.get("destination_place_id") != destination.get("place_id"):
                return True
            if _optional_int(route.get("duration_seconds")) is None:
                return True
            if _optional_float(route.get("distance_meters")) is None:
                return True
    return False


def rank_nearby_candidates(
    candidates: list[dict[str, Any]],
    *,
    anchor: Mapping[str, Any] | None,
    city: str,
    entity_type: str | None,
    excluded_place_ids: set[str],
) -> list[dict[str, Any]]:
    """Shortlist by geometry; the traffic provider remains route authority."""

    anchor_coordinates = _coordinates(anchor or {})
    ranked: list[dict[str, Any]] = []
    for item in candidates:
        place_id = str(item.get("place_id") or "")
        if not place_id or place_id in excluded_place_ids:
            continue
        if city and normalize_text(str(item.get("city") or "")) != normalize_text(city):
            continue
        if entity_type and str(item.get("entity_type") or "") != entity_type:
            continue
        candidate = deepcopy(item)
        coordinates = _coordinates(candidate)
        if anchor_coordinates is not None and coordinates is not None:
            candidate["distance_km"] = round(
                _haversine_km(anchor_coordinates, coordinates),
                2,
            )
        ranked.append(candidate)
    return sorted(
        ranked,
        key=lambda item: (
            _optional_float(item.get("distance_km")) is None,
            _optional_float(item.get("distance_km")) or math.inf,
            -(_optional_float(item.get("score")) or 0),
            str(item.get("place_id")),
        ),
    )


def _coordinates(value: Mapping[str, Any]) -> tuple[float, float] | None:
    attributes = value.get("attributes")
    attrs = attributes if isinstance(attributes, Mapping) else {}
    latitude = _optional_float(attrs.get("latitude", attrs.get("lat")))
    longitude = _optional_float(attrs.get("longitude", attrs.get("lng")))
    if latitude is None or longitude is None:
        return None
    return latitude, longitude


def _haversine_km(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> float:
    lat1, lng1 = map(math.radians, origin)
    lat2, lng2 = map(math.radians, destination)
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    )
    return 6371.0088 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "ActiveTripPlan",
    "BudgetSummary",
    "PlanChange",
    "PlanMutation",
    "PlanOperation",
    "active_plan_from_value",
    "annotate_itinerary_costs",
    "apply_plan_mutation",
    "build_active_trip_plan",
    "choose_add_candidate",
    "compact_active_plan_context",
    "extract_budget_vnd",
    "extract_occupancy",
    "invalidate_adjacent_routes",
    "locate_target_slot",
    "normalize_transport_units",
    "rank_nearby_candidates",
    "resolve_plan_mutation",
]
