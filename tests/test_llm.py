from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.core_ai.nextrip_agent.conversation import ConversationResolution
from src.infra.llm import (
    GeminiAnswerGenerator,
    GeminiConversationContextualizer,
    _compact_hotel_availability,
    _ensure_single_entity_reference,
    _guard_presentation_output,
    _protected_context,
    _repair_missing_fact_references,
    _restore_references,
)
from src.shared import telemetry


def test_compact_hotel_availability_exposes_stale_price_fallback() -> None:
    compact = _compact_hotel_availability(
        {
            "selected_window_index": None,
            "windows": [
                {
                    "check_in": "2026-08-29",
                    "check_out": "2026-08-30",
                    "lookup_status": "stale",
                    "availability": "available",
                    "latest_observed_at": "2026-08-28T14:41:41Z",
                    "offers": [
                        {
                            "seller": "Booking.com",
                            "currency": "VND",
                            "nightly_amount": "882000",
                            "observed_at": "2026-08-28T14:41:41Z",
                            "stale_after": "2026-08-28T19:41:41Z",
                            "stale": True,
                        }
                    ],
                }
            ],
        }
    )

    assert compact is not None
    assert compact["selected_window_index"] is None
    assert compact["price_window_index"] == 0
    assert compact["using_stale_fallback"] is True
    assert compact["selected_window"]["offers"][0]["nightly_amount"] == "882000"
    assert compact["selected_window"]["offers"][0]["stale"] is True


def test_grounded_context_protects_entity_names_and_fact_values() -> None:
    context, replacements = _protected_context(
        question="Mr. Mộc ở đâu?",
        answer_type="entity_detail",
        evidence=[{
            "place_id": "rest_qn_074",
            "name": "Hải sản tươi sống Mr. Mộc",
            "city": "Quy Nhơn",
            "entity_type": "restaurant",
            "category": "Hải sản",
        }],
        facts=[{
            "subject_id": "rest_qn_074",
            "predicate": "address",
            "value": "56 Nguyễn Thị Định",
            "unit": None,
        }],
        matched_paths=[],
    )

    assert context["retrieved_places"][0]["reference"] == "[[PLACE_1]]"
    assert context["retrieved_places"][0]["city"] == "[[CITY_1]]"
    assert "Hải sản tươi sống Mr. Mộc" not in str(context)
    assert replacements["[[FACT_1]]"] == "56 Nguyễn Thị Định"
    assert _restore_references(
        "[[PLACE_1]] ở [[CITY_1]] có địa chỉ [[FACT_1]].",
        replacements,
    ) == "Hải sản tươi sống Mr. Mộc ở Quy Nhơn có địa chỉ 56 Nguyễn Thị Định."


def test_reference_restoration_removes_adjacent_grounded_name_duplicates() -> None:
    answer = _restore_references(
        "[[PLACE_1]] thuộc [[CITY_1]] Quy Nhơn.",
        {
            "[[PLACE_1]]": "Eo Gió",
            "[[CITY_1]]": "Quy Nhơn",
        },
    )

    assert answer == "Eo Gió thuộc Quy Nhơn."


def test_reference_restoration_deduplicates_single_place_fact_bullets() -> None:
    answer = _restore_references(
        "[[PLACE_1]]\n- address: 26 To Hien Thanh [[PLACE_1]]\n- description: [[PLACE_1]] serves seafood.",
        {"[[PLACE_1]]": "Moc Quan"},
    )

    assert answer.count("Moc Quan") == 1


def test_grounded_context_masks_place_name_in_question() -> None:
    context, _ = _protected_context(
        question="Khách sạn An House địa chỉ ở đâu?",
        answer_type="entity_detail",
        evidence=[{
            "place_id": "hotel_qn_001",
            "name": "Khách sạn An House",
            "city": "Quy Nhơn",
            "entity_type": "hotel",
            "category": "hotel",
        }],
        facts=[],
        matched_paths=[],
    )

    assert context["question"] == "[[PLACE_1]] địa chỉ ở đâu?"


def test_grounded_context_uses_structured_itinerary_place_references() -> None:
    context, replacements = _protected_context(
        question="Lên lịch trình một ngày",
        answer_type="recommendation",
        evidence=[{
            "place_id": "attr_qn_001",
            "name": "Eo Gió",
            "city": "Quy Nhơn",
            "entity_type": "attraction",
        }],
        facts=[],
        matched_paths=[],
        itinerary=[{
            "day": 1,
            "slots": [{
                "order": 1,
                "start_time": "09:00",
                "end_time": "10:30",
                "place_id": "attr_qn_001",
                "entity_type": "attraction",
            }],
        }],
    )

    assert context["structured_itinerary"][0]["slots"][0]["place"] == "[[PLACE_1]]"
    assert replacements["[[PLACE_1]]"] == "Eo Gió"


def test_grounded_context_formats_list_fact_as_readable_text() -> None:
    _, replacements = _protected_context(
        question="Món đặc trưng là gì?",
        answer_type="entity_detail",
        evidence=[],
        facts=[{
            "subject_id": "rest_qn_074",
            "predicate": "signature_dishes",
            "value": ["ghẹ rang me", "tôm nướng"],
            "unit": None,
        }],
        matched_paths=[],
    )

    assert replacements["[[FACT_1]]"] == "ghẹ rang me, tôm nướng"


def test_entity_profile_adds_verified_place_heading_when_model_omits_it() -> None:
    answer = _ensure_single_entity_reference(
        "- Địa chỉ: [[FACT_1]]",
        answer_type="entity_detail",
        evidence=[{"place_id": "rest_qn_074"}],
    )

    assert answer == "[[PLACE_1]]\n- Địa chỉ: [[FACT_1]]"


def test_entity_profile_preserves_grounded_repeated_place_reference() -> None:
    answer = _ensure_single_entity_reference(
        "[[PLACE_1]] là nhà hàng. Địa chỉ [[PLACE_1]]: [[FACT_1]]",
        answer_type="entity_detail",
        evidence=[{"place_id": "rest_qn_074"}],
    )

    assert answer.count("[[PLACE_1]]") == 2


def test_entity_profile_does_not_hide_empty_model_response() -> None:
    assert _ensure_single_entity_reference(
        "",
        answer_type="entity_detail",
        evidence=[{"place_id": "rest_qn_074"}],
    ) == ""


def test_grounded_answer_rejects_missing_reference() -> None:
    with pytest.raises(RuntimeError, match="grounded reference contract"):
        _restore_references("Một câu trả lời không có tên.", {"[[PLACE_1]]": "Mr. Mộc"})


def test_missing_fact_references_are_appended_from_verified_context() -> None:
    repaired = _repair_missing_fact_references(
        "[[PLACE_1]] là một khách sạn ven biển.",
        {
            "verified_facts": [
                {
                    "reference": "[[FACT_1]]",
                    "predicate": "address",
                },
                {
                    "reference": "[[FACT_5]]",
                    "predicate": "price",
                },
            ]
        },
    )

    assert repaired.count("[[FACT_1]]") == 1
    assert repaired.count("[[FACT_5]]") == 1
    assert "- Địa chỉ: [[FACT_1]]" in repaired
    assert "- Giá: [[FACT_5]]" in repaired
    assert _restore_references(
        repaired,
        {
            "[[PLACE_1]]": "Hotel Seagull",
            "[[FACT_1]]": "489 An Dương Vương",
            "[[FACT_5]]": "882.000 VND",
        },
    ).endswith("- Giá: 882.000 VND")


def test_grounded_answer_rejects_unknown_reference() -> None:
    with pytest.raises(RuntimeError, match="unknown grounded references"):
        _restore_references("[[PLACE_1]] ở [[CITY_9]].", {"[[PLACE_1]]": "Mr. Mộc"})


def test_context_and_answer_use_separate_models_with_minimal_thinking() -> None:
    app_settings = Settings(
        _env_file=None,
        google_api_key="test-key",
        gemini_context_model="test-context-model",
        gemini_answer_model="test-answer-model",
        gemini_thinking_level="minimal",
    )
    context_calls = []
    answer_calls = []
    contextualizer = GeminiConversationContextualizer(app_settings)
    answer_generator = GeminiAnswerGenerator(app_settings)
    contextualizer._client.close()
    answer_generator._client.close()
    contextualizer._client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: (
                context_calls.append(kwargs)
                or SimpleNamespace(
                    parsed=ConversationResolution(
                        route="travel",
                        standalone_message="standalone",
                        confidence=1,
                    ),
                    text=None,
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=10,
                        candidates_token_count=5,
                        thoughts_token_count=2,
                    ),
                )
            )
        )
    )
    answer_generator._client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: (
                answer_calls.append(kwargs)
                or SimpleNamespace(
                    text="OK",
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=20,
                        candidates_token_count=4,
                        thoughts_token_count=3,
                    ),
                )
            )
        )
    )

    contextualizer.contextualize(
        message="follow-up",
        history=[{"role": "user", "content": "first turn"}],
        prior_summary=None,
        structured_context={},
    )
    answer_generator.generate(
        question="question",
        answer_type="aggregate_count",
        evidence=[],
        facts=[],
        matched_paths=[],
    )

    assert context_calls[0]["model"] == "test-context-model"
    assert answer_calls[0]["model"] == "test-answer-model"
    assert (
        context_calls[0]["config"].thinking_config.thinking_level.value
        == "MINIMAL"
    )
    assert answer_calls[0]["config"].thinking_config.thinking_level.value == "MINIMAL"


def test_telemetry_bills_thinking_tokens_as_output(monkeypatch) -> None:
    token_records = []
    cost_records = []
    monkeypatch.setattr(
        telemetry,
        "_llm_tokens",
        SimpleNamespace(
            add=lambda value, attributes: token_records.append((value, attributes))
        ),
    )
    monkeypatch.setattr(
        telemetry,
        "_llm_cost",
        SimpleNamespace(
            add=lambda value, attributes: cost_records.append((value, attributes))
        ),
    )

    telemetry.record_llm_usage(
        "test-model",
        100,
        20,
        thinking_tokens=30,
        input_cost_per_million=1.5,
        output_cost_per_million=7.5,
    )

    assert token_records[-1] == (
        30,
        {"model": "test-model", "direction": "thinking"},
    )
    assert cost_records == [(0.000525, {"model": "test-model"})]


def test_grounded_context_exposes_only_sanitized_presentation_attributes() -> None:
    context, _ = _protected_context(
        question="Khach san co tien ich gi?",
        answer_type="entity_detail",
        evidence=[{
            "place_id": "hotel_qn_001",
            "name": "Hotel Test",
            "city": "Quy Nhon",
            "entity_type": "hotel",
            "category": "hotel",
            "attributes": {
                "address": "01 Test",
                "amenities": [
                    "beach_access",
                    "rocky_beach",
                    "wifi mien phi",
                    "unknown_internal_slug",
                ],
                "is_indoor": True,
                "source_file": "hotel_final.json",
                "embedding_text": "private retrieval text",
                "unknown_boolean": False,
            },
        }],
        facts=[],
        matched_paths=[],
    )

    attributes = context["retrieved_places"][0]["attributes"]
    serialized = json.dumps(context, ensure_ascii=False).casefold()
    assert attributes == {
        "address": "01 Test",
        "amenities": ["lối ra biển", "bãi đá ven biển", "wifi mien phi"],
        "space": "trong nhà",
    }
    assert "beach_access" not in serialized
    assert "rocky_beach" not in serialized
    assert "unknown_internal_slug" not in serialized
    assert "source_file" not in serialized
    assert "embedding_text" not in serialized
    assert "true" not in serialized
    assert "false" not in serialized


def test_final_output_guard_translates_known_tokens_and_raw_boolean() -> None:
    assert _guard_presentation_output(
        "Khach san co beach_access va rocky_beach."
    ) == "Khach san co lối ra biển va bãi đá ven biển."
    assert _guard_presentation_output("is_indoor: true") == "không gian trong nhà: có"


def test_grounded_answer_allows_omitted_city_reference() -> None:
    assert _restore_references(
        "Gợi ý [[PLACE_1]].",
        {"[[PLACE_1]]": "Mr. Mộc", "[[CITY_1]]": "Quy Nhơn"},
    ) == "Gợi ý Mr. Mộc."
