from __future__ import annotations

from src.apis.domains.chat.schemas import ChatRequest
from src.apis.domains.chat.service import handle_chat
from src.core_ai.nextrip_agent.conversation import (
    contextualize_message,
    resolve_conversation_context,
)
from src.infra.chat_store import InMemoryChatStore
from src.infra.weather import DailyForecast


class FailingKbClient:
    def query_typed(self, **kwargs):
        raise AssertionError("Weather-only follow-up must not call GraphRAG")


class FakeWeatherClient:
    configured = True

    def forecast(self, latitude, longitude, target_date, today):
        return DailyForecast(
            forecast_date=target_date,
            condition="Có mây",
            condition_type="PARTLY_CLOUDY",
            min_temperature_c=25,
            max_temperature_c=31,
            precipitation_probability=20,
            thunderstorm_probability=None,
            uv_index=6,
            wind_gust_kph=18,
        )


class FoodConversationKbClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query_typed(self, *, query, top_k, kb_version="v5"):
        self.queries.append(query)
        if len(self.queries) == 1:
            return {
                "kb_version": kb_version,
                "answer_type": "entity_list",
                "targets": [
                    {
                        "target_id": "concept:dish:banh-hoi",
                        "kind": "dish",
                        "name": "bánh hỏi",
                        "score": 2.0,
                    },
                    {
                        "target_id": "concept:dish:bun-ca",
                        "kind": "dish",
                        "name": "bún cá",
                        "score": 1.0,
                    },
                ],
                "query_plan": {
                    "targets": [{"kind": "dish", "value": None}],
                },
            }
        return {
            "kb_version": kb_version,
            "answer_type": "recommendation",
            "recommendations": [
                {
                    "place_id": "rest_qn_001",
                    "name": "Bánh hỏi Hồng Thanh",
                    "city": "Quy Nhơn",
                    "entity_type": "restaurant",
                    "attributes": {"address": "22 Phan Bội Châu, Quy Nhơn"},
                }
            ],
            "query_plan": {
                "targets": [
                    {"kind": "dish", "value": "bánh hỏi"},
                    {"kind": "dish", "value": "bún cá"},
                ],
            },
        }


class GroundedAnswerGenerator:
    def generate(self, **kwargs):
        return "Câu trả lời có nguồn."

    def synthesize(self, **kwargs):
        return self.generate(**kwargs)


def test_context_resolver_prefers_explicit_city() -> None:
    context = resolve_conversation_context(
        message="Thời tiết ngày mai thế nào?",
        explicit_city="Đà Nẵng",
        history=[{"role": "user", "content": "Tôi đang ở Quy Nhơn"}],
    )

    assert context.city == "Đà Nẵng"
    assert context.city_source == "request"


def test_context_resolver_prefers_city_named_in_current_message() -> None:
    context = resolve_conversation_context(
        message="Lên lịch trình một ngày khám phá Quy Nhơn",
        explicit_city="Đà Nẵng",
        history=[],
    )

    assert context.city == "Quy Nhơn"
    assert context.city_source == "current_message"


def test_weather_follow_up_inherits_city_from_session() -> None:
    store = InMemoryChatStore()
    store.save_message(
        "multi-turn",
        "previous-user",
        "user",
        "Trời mưa thì nên đi đâu ở Quy Nhơn?",
    )

    response = handle_chat(
        ChatRequest(
            message="Thời tiết ngày mai như thế nào?",
            session_id="multi-turn",
            kb_version="v3",
        ),
        FailingKbClient(),
        weather_client=FakeWeatherClient(),
        chat_store=store,
    )

    assert response.orchestration_mode == "weather_only"
    assert response.resolved_context == {
        "city": "Quy Nhơn",
        "city_source": "conversation_history",
        "travel_date": None,
        "travel_date_source": None,
        "recent_place_ids": [],
        "recent_dishes": [],
    }
    assert response.weather is not None
    assert response.weather.location == "Quy Nhơn"
    assert response.required_tools == []
    assert response.trace[0]["node"] == "context_resolver"
    assert all(event.get("node") for event in response.trace)


def test_weather_without_location_returns_clarification() -> None:
    response = handle_chat(
        ChatRequest(
            message="Thời tiết ngày mai như thế nào?",
            session_id="new-session",
            kb_version="v3",
        ),
        FailingKbClient(),
        weather_client=FakeWeatherClient(),
        chat_store=InMemoryChatStore(),
    )

    assert response.weather is None
    assert response.missing_fields == ["city"]
    assert response.answer == "Bạn muốn xem thời tiết ở Quy Nhơn hay Đà Nẵng?"
    assert any(event.get("status") == "needs_input" for event in response.trace)


def test_food_follow_up_expands_recent_dishes_from_assistant_metadata() -> None:
    context = resolve_conversation_context(
        message="Cho tôi địa chỉ các quán bán các món này",
        explicit_city=None,
        history=[
            {"role": "user", "content": "Ở Quy Nhơn có món gì ngon?"},
            {
                "role": "assistant",
                "content": "Bánh hỏi và bún cá.",
                "city": "Quy Nhơn",
                "metadata": {
                    "referenced_entities": [
                        {"id": "dish:banh-hoi", "name": "bánh hỏi", "kind": "dish"},
                        {"id": "dish:bun-ca", "name": "bún cá", "kind": "dish"},
                    ]
                },
            },
        ],
    )

    query = contextualize_message(
        "Cho tôi địa chỉ các quán bán các món này",
        context,
    )

    assert context.recent_dishes == ("bánh hỏi", "bún cá")
    assert "bánh hỏi, bún cá" in query
    assert context.city == "Quy Nhơn"


def test_food_follow_up_is_contextualized_across_chat_turns() -> None:
    store = InMemoryChatStore()
    kb_client = FoodConversationKbClient()
    generator = GroundedAnswerGenerator()
    session_id = "food-follow-up"

    first = handle_chat(
        ChatRequest(
            message="Ở Quy Nhơn có món gì ngon?",
            session_id=session_id,
            kb_version="v5",
        ),
        kb_client,
        generator,
        chat_store=store,
    )
    second = handle_chat(
        ChatRequest(
            message="Cho tôi các địa chỉ quán bán các món này",
            session_id=session_id,
            kb_version="v5",
        ),
        kb_client,
        generator,
        chat_store=store,
    )

    assert [item.entity_type for item in first.evidence] == ["dish", "dish"]
    assert "Các món được nhắc đến ở lượt trước: bánh hỏi, bún cá" in kb_client.queries[1]
    assert second.resolved_context["recent_dishes"] == ["bánh hỏi", "bún cá"]
    assert second.evidence[0].attributes["address"].startswith("22 Phan Bội Châu")
    assert second.missing_fields == []
