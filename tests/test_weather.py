from __future__ import annotations

from datetime import date

import httpx

from src.core_ai.nextrip_agent.weather import WeatherAgent, assess_suitability
from src.infra.weather import OpenMeteoWeatherClient


def _weather_payload(today: date) -> dict:
    return {
        "daily": {
            "time": [today.isoformat()],
            "weather_code": [2],
            "temperature_2m_min": [25],
            "temperature_2m_max": [31],
            "precipitation_probability_max": [20],
            "uv_index_max": [6],
            "wind_gusts_10m_max": [18],
        }
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
    client = OpenMeteoWeatherClient(client=httpx.Client(transport=transport))

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

    assert result.condition == "Có mây rải rác"
    assert result.precipitation_probability == 20
    assert result.suitability == "suitable"
    assert result.source == "Open-Meteo"


def test_heavy_rain_is_unsuitable() -> None:
    suitability, _ = assess_suitability(
        precipitation=75,
        thunderstorm=10,
        uv_index=2,
        wind_gust=15,
        max_temperature=29,
    )

    assert suitability == "unsuitable"
