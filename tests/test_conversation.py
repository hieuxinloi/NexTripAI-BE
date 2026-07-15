from __future__ import annotations

from src.apis.domains.chat.schemas import ChatRequest
from src.apis.domains.chat.service import handle_chat
from src.core_ai.nextrip_agent.conversation import resolve_conversation_context
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


def test_context_resolver_prefers_explicit_city() -> None:
    context = resolve_conversation_context(
        message="Thời tiết ngày mai thế nào?",
        explicit_city="Đà Nẵng",
        history=[{"role": "user", "content": "Tôi đang ở Quy Nhơn"}],
    )

    assert context.city == "Đà Nẵng"
    assert context.city_source == "request"


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
