from datetime import date

from src.core_ai.nextrip_agent.trip_plan import (
    PlanMutation,
    PlanOperation,
    apply_plan_mutation,
    build_active_trip_plan,
    invalidate_adjacent_routes,
    rank_nearby_candidates,
    resolve_plan_mutation,
)


def _itinerary():
    return [
        {
            "day": 1,
            "slots": [
                {
                    "order": 1,
                    "start_time": "09:00",
                    "end_time": "10:30",
                    "place_id": "attr_qn_001",
                    "name": "Bảo tàng A",
                    "city": "Quy Nhơn",
                    "entity_type": "attraction",
                    "role": "activity",
                    "transport_to_next": {
                        "status": "recommended",
                        "origin_place_id": "attr_qn_001",
                        "destination_place_id": "rest_qn_001",
                        "departure_time": "2026-09-01T10:30:00+07:00",
                        "recommended_mode": "two_wheeler",
                        "distance_meters": 7400,
                        "duration_seconds": 1080,
                    },
                },
                {
                    "order": 2,
                    "start_time": "11:00",
                    "end_time": "12:00",
                    "place_id": "rest_qn_001",
                    "name": "Quán ăn B",
                    "city": "Quy Nhơn",
                    "entity_type": "restaurant",
                    "role": "meal",
                },
            ],
        }
    ]


def _evidence():
    return [
        {
            "place_id": "attr_qn_001",
            "name": "Bảo tàng A",
            "city": "Quy Nhơn",
            "entity_type": "attraction",
            "attributes": {
                "lat": 13.77,
                "lng": 109.22,
                "ticket_price_adult": 50_000,
            },
        },
        {
            "place_id": "rest_qn_001",
            "name": "Quán ăn B",
            "city": "Quy Nhơn",
            "entity_type": "restaurant",
            "attributes": {"lat": 13.78, "lng": 109.23},
        },
    ]


def test_plan_has_stable_slots_route_units_and_partial_grounded_budget():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhơn",
        start_date=date(2026, 9, 1),
        duration_days=1,
        operation=PlanOperation.CREATE,
        mutation=PlanMutation(budget_vnd=500_000, adults=2, rooms=1),
    )

    first, second = plan.itinerary[0]["slots"]
    assert first["slot_id"].endswith(":d1:s1")
    assert first["transport_to_next"]["duration_minutes"] == 18
    assert first["transport_to_next"]["distance_km"] == 7.4
    assert first["transport_to_next"]["mode_label"] == "Xe máy"
    assert first["cost_estimate"]["amount_min"] == 100_000
    assert second["cost_estimate"]["status"] == "unknown"
    assert plan.budget_summary is not None
    assert plan.budget_summary.status == "partial"
    assert plan.budget_summary.estimated_min == 100_000
    assert plan.budget_summary.within_budget is None
    assert plan.budget_summary.missing_place_ids == ["rest_qn_001"]


def test_replace_preserves_slot_id_and_only_invalidates_adjacent_legs():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhơn",
        start_date=date(2026, 9, 1),
        duration_days=1,
        operation=PlanOperation.CREATE,
    )
    target_id = plan.itinerary[0]["slots"][1]["slot_id"]
    mutation = PlanMutation(
        operation=PlanOperation.REPLACE_SLOT,
        expected_revision=1,
        target_slot_id=target_id,
    )
    candidate = {
        "place_id": "rest_qn_002",
        "name": "Quán ăn C",
        "city": "Quy Nhơn",
        "entity_type": "restaurant",
        "attributes": {"price_per_person_min": 80_000, "price_per_person_max": 120_000},
    }

    days, _, change = apply_plan_mutation(plan, mutation, [candidate])
    days = invalidate_adjacent_routes(days, change.changed_slot_ids)

    assert days[0]["slots"][1]["slot_id"] == target_id
    assert days[0]["slots"][1]["place_id"] == "rest_qn_002"
    assert days[0]["slots"][0]["transport_to_next"] is None
    assert change.preserved_slot_ids == [plan.itinerary[0]["slots"][0]["slot_id"]]


def test_suggest_nearby_is_non_mutating_and_candidates_are_coordinate_ranked():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhơn",
        start_date=None,
        duration_days=1,
        operation=PlanOperation.CREATE,
    )
    mutation = resolve_plan_mutation(
        "Gợi ý quán ăn lân cận Bảo tàng A",
        plan,
    )
    assert mutation.operation is PlanOperation.SUGGEST_NEARBY

    ranked = rank_nearby_candidates(
        [
            {
                "place_id": "rest_qn_far",
                "city": "Quy Nhơn",
                "entity_type": "restaurant",
                "attributes": {"lat": 13.9, "lng": 109.3},
            },
            {
                "place_id": "rest_qn_near",
                "city": "Quy Nhơn",
                "entity_type": "restaurant",
                "attributes": {"lat": 13.771, "lng": 109.221},
            },
        ],
        anchor=plan.selected_places[0],
        city="Quy Nhơn",
        entity_type="restaurant",
        excluded_place_ids={"attr_qn_001", "rest_qn_001"},
    )
    assert [item["place_id"] for item in ranked] == ["rest_qn_near", "rest_qn_far"]
    assert ranked[0]["distance_km"] < ranked[1]["distance_km"]


def test_occupancy_and_budget_are_extracted_without_changing_default_behavior():
    mutation = resolve_plan_mutation(
        "Lên lịch 3 ngày cho 4 người lớn, 2 phòng, budget 8 triệu",
        None,
    )

    assert mutation.operation is PlanOperation.UPDATE_CONSTRAINTS
    assert mutation.adults == 4
    assert mutation.rooms == 2
    assert mutation.budget_vnd == 8_000_000


def test_vietnamese_add_intent_is_parsed_without_inventing_a_place():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhon",
        start_date=None,
        duration_days=1,
        operation=PlanOperation.CREATE,
    )

    mutation = resolve_plan_mutation(
        "Them Quan C vao ngay 1 luc 15:00",
        plan,
    )

    assert mutation.operation is PlanOperation.ADD_SLOT
    assert mutation.target_day == 1
    assert mutation.replacement_place_name == "quan c"
    assert mutation.start_time == "15:00"
    assert mutation.expected_revision == plan.revision


def test_add_slot_uses_grounded_candidate_preserves_ids_and_invalidates_routes():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhon",
        start_date=None,
        duration_days=1,
        operation=PlanOperation.CREATE,
    )
    original_ids = [slot["slot_id"] for slot in plan.itinerary[0]["slots"]]
    mutation = PlanMutation(
        operation=PlanOperation.ADD_SLOT,
        expected_revision=1,
        target_day=1,
        target_order=2,
        replacement_place_name="Quan C",
        start_time="10:40",
        end_time="11:10",
    )
    candidates = [
        {
            "place_id": "cafe_qn_other",
            "name": "Other Cafe",
            "city": "Quy Nhon",
            "entity_type": "cafe",
        },
        {
            "place_id": "cafe_qn_003",
            "name": "Quan C",
            "city": "Quy Nhon",
            "entity_type": "cafe",
            "attributes": {"lat": 13.78, "lng": 109.23},
        },
    ]

    days, selected, change = apply_plan_mutation(plan, mutation, candidates)
    days = invalidate_adjacent_routes(days, change.changed_slot_ids)
    slots = days[0]["slots"]

    assert [slot["order"] for slot in slots] == [1, 2, 3]
    assert slots[0]["slot_id"] == original_ids[0]
    assert slots[2]["slot_id"] == original_ids[1]
    assert slots[1]["place_id"] == "cafe_qn_003"
    assert slots[1]["slot_id"].startswith(f"{plan.plan_id}:slot:")
    assert slots[0]["transport_to_next"] is None
    assert slots[1]["transport_to_next"] is None
    assert change.changed_slot_ids == [slots[1]["slot_id"]]
    assert change.preserved_slot_ids == original_ids
    assert any(item["place_id"] == "cafe_qn_003" for item in selected)


def test_add_slot_fails_when_requested_place_is_not_in_candidates():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhon",
        start_date=None,
        duration_days=1,
        operation=PlanOperation.CREATE,
    )
    mutation = PlanMutation(
        operation=PlanOperation.ADD_SLOT,
        target_day=1,
        replacement_place_name="Not In Retrieval",
    )

    try:
        apply_plan_mutation(
            plan,
            mutation,
            [{"place_id": "cafe_real", "name": "Grounded Cafe"}],
        )
    except ValueError as exc:
        assert str(exc) == "plan_add_candidate_not_found"
    else:
        raise AssertionError("An ungrounded requested place must not be added")


def test_move_slot_keeps_identity_renumbers_days_and_invalidates_old_and_new_legs():
    itinerary = _itinerary() + [
        {
            "day": 2,
            "slots": [
                {
                    "order": 1,
                    "start_time": "09:00",
                    "end_time": "10:00",
                    "place_id": "attr_qn_002",
                    "name": "Attraction D",
                    "city": "Quy Nhon",
                    "entity_type": "attraction",
                    "role": "activity",
                    "transport_to_next": {"status": "recommended"},
                }
            ],
        }
    ]
    plan = build_active_trip_plan(
        itinerary=itinerary,
        evidence=_evidence()
        + [
            {
                "place_id": "attr_qn_002",
                "name": "Attraction D",
                "city": "Quy Nhon",
                "entity_type": "attraction",
            }
        ],
        city="Quy Nhon",
        start_date=None,
        duration_days=2,
        operation=PlanOperation.CREATE,
    )
    source_slots = plan.itinerary[0]["slots"]
    moved_id = source_slots[1]["slot_id"]
    old_previous_id = source_slots[0]["slot_id"]
    mutation = PlanMutation(
        operation=PlanOperation.MOVE_SLOT,
        target_slot_id=moved_id,
        destination_day=2,
        destination_order=1,
    )

    days, _, change = apply_plan_mutation(plan, mutation, [])
    days = invalidate_adjacent_routes(days, change.changed_slot_ids)

    assert [slot["order"] for slot in days[0]["slots"]] == [1]
    assert [slot["order"] for slot in days[1]["slots"]] == [1, 2]
    assert days[1]["slots"][0]["slot_id"] == moved_id
    assert days[0]["slots"][0]["slot_id"] == old_previous_id
    assert days[0]["slots"][0]["transport_to_next"] is None
    assert days[1]["slots"][0]["transport_to_next"] is None
    assert change.changed_slot_ids == [moved_id, old_previous_id]


def test_update_time_preserves_duration_and_invalidates_adjacent_routes():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhon",
        start_date=None,
        duration_days=1,
        operation=PlanOperation.CREATE,
    )
    target_id = plan.itinerary[0]["slots"][1]["slot_id"]
    parsed = resolve_plan_mutation("Doi gio Quan an B sang luc 15h", plan)
    assert parsed.operation is PlanOperation.UPDATE_TIME
    assert parsed.start_time == "15:00"

    mutation = parsed.model_copy(update={"target_slot_id": target_id})
    days, _, change = apply_plan_mutation(plan, mutation, [])
    days = invalidate_adjacent_routes(days, change.changed_slot_ids)

    target = days[0]["slots"][1]
    assert target["slot_id"] == target_id
    assert target["start_time"] == "15:00"
    assert target["end_time"] == "16:00"
    assert days[0]["slots"][0]["transport_to_next"] is None
    assert change.changed_slot_ids == [target_id]


def test_replace_intent_keeps_the_requested_destination_name() -> None:
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhơn",
        start_date=None,
        duration_days=1,
        operation=PlanOperation.CREATE,
    )

    mutation = resolve_plan_mutation(
        "Thay Quán ăn B bằng Nhà hàng Hàn Quốc C",
        plan,
    )

    assert mutation.operation is PlanOperation.REPLACE_SLOT
    assert mutation.target_place_name == "Quán ăn B"
    assert mutation.replacement_place_name == "nha hang han quoc c"


def test_vietnamese_move_intent_separates_source_and_destination():
    plan = build_active_trip_plan(
        itinerary=_itinerary(),
        evidence=_evidence(),
        city="Quy Nhon",
        start_date=None,
        duration_days=2,
        operation=PlanOperation.CREATE,
    )

    mutation = resolve_plan_mutation(
        "Chuyen diem thu 2 tu ngay 1 sang ngay 2 vi tri 1",
        plan,
    )

    assert mutation.operation is PlanOperation.MOVE_SLOT
    assert mutation.target_day == 1
    assert mutation.target_order == 2
    assert mutation.destination_day == 2
    assert mutation.destination_order == 1
