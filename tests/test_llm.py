from __future__ import annotations

import pytest

from src.infra.llm import (
    _ensure_single_entity_reference,
    _protected_context,
    _restore_references,
)


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


def test_grounded_answer_rejects_unknown_reference() -> None:
    with pytest.raises(RuntimeError, match="unknown grounded references"):
        _restore_references("[[PLACE_1]] ở [[CITY_9]].", {"[[PLACE_1]]": "Mr. Mộc"})


def test_grounded_answer_allows_omitted_city_reference() -> None:
    assert _restore_references(
        "Gợi ý [[PLACE_1]].",
        {"[[PLACE_1]]": "Mr. Mộc", "[[CITY_1]]": "Quy Nhơn"},
    ) == "Gợi ý Mr. Mộc."
