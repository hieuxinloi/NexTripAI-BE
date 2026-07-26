from src.core_ai.nextrip_agent.nodes.knowledge import _query_with_city


def test_query_with_city_does_not_duplicate_accentless_city() -> None:
    query = "Goi y dia diem o Da Nang"

    assert _query_with_city(query, "Đà Nẵng") == query


def test_query_with_city_adds_missing_context_city() -> None:
    assert _query_with_city("Gợi ý quán cà phê", "Quy Nhơn") == (
        "Gợi ý quán cà phê o Quy Nhơn"
    )


def test_query_with_city_does_not_override_another_explicit_city() -> None:
    assert _query_with_city("5 nhà hàng ở Quy Nhơn", "Đà Nẵng") == (
        "5 nhà hàng ở Quy Nhơn"
    )
