from src.core_ai.nextrip_agent.nodes.answer import (
    _clarification_answer,
    _display_missing_fields,
)


def test_missing_fields_are_rendered_as_user_facing_labels() -> None:
    assert _display_missing_fields(
        ["query_constraints", "distance_between:Quy Nhơn:Đà Nẵng"]
    ) == [
        "tiêu chí hoặc điều kiện ưu tiên",
        "khoảng cách giữa Quy Nhơn và Đà Nẵng",
    ]


def test_missing_city_uses_destination_choice_question() -> None:
    assert _clarification_answer(["city"]) == (
        "Bạn muốn đi Quy Nhơn, Đà Nẵng hay xem gợi ý ở cả hai thành phố?"
    )
