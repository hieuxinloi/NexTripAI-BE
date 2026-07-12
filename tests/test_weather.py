from __future__ import annotations

from datetime import date

import httpx

from src.core_ai.nextrip_agent.weather import WeatherAgent, assess_suitability
from src.infra.weather import GoogleWeatherClient


def _weather_payload(today: date) -> dict:
    return {
        "forecastDays": [
            {
                "displayDate": {
                    "year": today.year,
                    "month": today.month,
                    "day": today.day,
                },
                "daytimeForecast": {
                    "weatherCondition": {
                        "type": "PARTLY_CLOUDY",
                        "description": {"text": "Có mây"},
                    },
                    "precipitation": {"probability": {"percent": 20}},
                    "thunderstormProbability": 5,
                    "uvIndex": 6,
                    "wind": {"gust": {"value": 18}},
                },
                "minTemperature": {"degrees": 25},
                "maxTemperature": {"degrees": 31},
            }
        ]
    }


def test_weather_agent_routes_and_assesses_vietnamese_question() -> None:
    today = date(2026, 7, 12)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=_weather_payload(today),
            request=request,
        )
    )
    client = GoogleWeatherClient(
        "test-key",
        client=httpx.Client(transport=transport),
    )

    assert WeatherAgent.should_run(
        message="Hôm nay Đà Nẵng có mưa không?",
        travel_date=None,
        include_weather=None,
    )
    result = WeatherAgent(client).run(
        message="Thời tiết Đà Nẵng",
        city="Đà Nẵng",
        travel_date=today,
        latitude=None,
        longitude=None,
    )

    assert result.condition == "Có mây"
    assert result.precipitation_probability == 20
    assert result.suitability == "suitable"


def test_heavy_rain_is_unsuitable() -> None:
    suitability, _ = assess_suitability(
        precipitation=75,
        thunderstorm=10,
        uv_index=2,
        wind_gust=15,
        max_temperature=29,
    )

    assert suitability == "unsuitable"
