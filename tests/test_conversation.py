from __future__ import annotations

import pytest

from src.apis.domains.chat.schemas import ChatRequest
from src.apis.domains.chat.service import (
    KnowledgeBaseUnavailableError,
    _kb_conversation_context,
    handle_chat,
)
from src.core_ai.nextrip_agent.conversation import (
    ConversationContext,
    ConversationResolution,
    ResolvedTurn,
    resolve_conversation_context,
    resolve_turn,
)
from src.infra.chat_store import InMemoryChatStore
from src.infra.weather import DailyForecast


class FailingKbClient:
    def query_typed(self, **kwargs):
        raise AssertionError("Weather-only follow-up must not call GraphRAG")


def test_kb_context_preserves_city_provenance_and_resolved_reference() -> None:
    context = ConversationContext(
        city="Quy Nhon",
        city_source="conversation_history",
        history_messages=2,
    )
    resolved = ResolvedTurn(
        resolution=ConversationResolution(
            standalone_message="Tim Highlight Coffee o Quy Nhon",
        ),
        status="completed",
        reason="model",
        original_message="Highlight Coffee o dau",
    )

    payload = _kb_conversation_context(
        session_memory={},
        context=context,
        resolved_turn=resolved,
    )

    assert payload == {
        "cities": ["Quy Nhon"],
        "city_source": "conversation_history",
        "turn_count": 2,
        "resolved_query": "Tim Highlight Coffee o Quy Nhon",
    }


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

    def forecast_range(self, latitude, longitude, start_date, duration_days, today):
        return [
            self.forecast(latitude, longitude, start_date, today)
            for _ in range(duration_days)
        ]


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


class FoodFollowUpContextualizer:
    def contextualize(self, **kwargs):
        history = kwargs["history"]
        entities = history[-1]["metadata"]["referenced_entities"]
        dishes = ", ".join(item["name"] for item in entities)
        return ConversationResolution(
            route="travel",
            standalone_message=f"Tìm quán ở Quy Nhơn bán {dishes}",
            summary="Người dùng quan tâm đến các món đặc sản Quy Nhơn.",
            confidence=0.99,
        )


class ConversationRecallContextualizer:
    def contextualize(self, **kwargs):
        return ConversationResolution(
            route="conversation",
            standalone_message=kwargs["message"],
            direct_answer="Câu trước bạn đã hỏi: Quy Nhơn có món gì ngon?",
            summary="Người dùng hỏi về món ngon Quy Nhơn.",
            confidence=1,
        )


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
        "conversation_route": "travel",
        "contextualized": False,
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
    assert response.clarification is not None
    assert response.clarification.field == "city"
    assert [option.value for option in response.clarification.options] == [
        "Đà Nẵng",
        "Quy Nhơn",
        "all",
    ]
    assert any(event.get("status") == "needs_input" for event in response.trace)


def test_chat_persists_clarification_label_instead_of_internal_query() -> None:
    store = InMemoryChatStore()

    with pytest.raises(KnowledgeBaseUnavailableError):
        handle_chat(
            ChatRequest(
                message="Tìm quán cà phê ngon",
                display_message="Quy Nhơn",
                session_id="clarification-display",
                city="Quy Nhơn",
                kb_version="v3",
            ),
            FailingKbClient(),
            weather_client=FakeWeatherClient(),
            chat_store=store,
        )

    messages = store.recent_messages("clarification-display", 10)
    assert messages[0]["content"] == "Quy Nhơn"


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

    resolved = resolve_turn(
        message="Cho tôi địa chỉ các quán bán các món này",
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
        context=context,
        contextualizer=FoodFollowUpContextualizer(),
    )

    assert context.recent_dishes == ("bánh hỏi", "bún cá")
    assert "bánh hỏi, bún cá" in resolved.resolution.standalone_message
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
        conversation_contextualizer=FoodFollowUpContextualizer(),
    )

    assert [item.entity_type for item in first.evidence] == ["dish", "dish"]
    assert kb_client.queries[1] == "Tìm quán ở Quy Nhơn bán bánh hỏi, bún cá"
    assert second.resolved_context["recent_dishes"] == ["bánh hỏi", "bún cá"]
    assert second.resolved_context["contextualized"] is True
    assert second.evidence[0].attributes["address"].startswith("22 Phan Bội Châu")
    assert second.missing_fields == []


def test_conversation_recall_is_answered_from_history_without_kb_call() -> None:
    store = InMemoryChatStore()
    store.save_message(
        "recall",
        "previous-user",
        "user",
        "Quy Nhơn có món gì ngon?",
    )
    store.save_message(
        "recall",
        "previous-assistant",
        "assistant",
        "Bánh hỏi và bún cá.",
    )

    response = handle_chat(
        ChatRequest(
            message="Câu trước tôi đã hỏi gì?",
            session_id="recall",
            kb_version="v5",
        ),
        FailingKbClient(),
        chat_store=store,
        conversation_contextualizer=ConversationRecallContextualizer(),
    )

    assert response.intent == "conversation_memory"
    assert response.orchestration_mode == "conversation"
    assert response.answer == "Câu trước bạn đã hỏi: Quy Nhơn có món gì ngon?"
    assert response.evidence == []
    assert store.get_session_memory("recall")["summary"]
