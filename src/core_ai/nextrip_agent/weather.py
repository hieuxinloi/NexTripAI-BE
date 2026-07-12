from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import re
import unicodedata

from pydantic import BaseModel

from src.infra.weather import (
    OpenMeteoWeatherClient,
    WeatherLocationRequired,
    WeatherUnavailable,
)


VIETNAM_TIMEZONE = timezone(timedelta(hours=7))
CITY_COORDINATES = {
    "da nang": ("Đà Nẵng", 16.0544, 108.2022),
    "quy nhon": ("Quy Nhơn", 13.7820, 109.2190),
}
WEATHER_TERMS = {
    "thoi tiet",
    "troi mua",
    "troi nang",
    "mua khong",
    "du bao",
    "nhiet do",
    "ngay mai",
    "hom nay",
}


class WeatherAssessment(BaseModel):
    location: str
    forecast_date: date
    condition: str
    condition_type: str | None = None
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    precipitation_probability: int | None = None
    thunderstorm_probability: int | None = None
    uv_index: int | None = None
    wind_gust_kph: float | None = None
    suitability: str
    advice: str
    source: str = "Open-Meteo"
    source_url: str = "https://open-meteo.com/"


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def vietnam_today() -> date:
    return datetime.now(VIETNAM_TIMEZONE).date()


class WeatherAgent:
    def __init__(self, client: OpenMeteoWeatherClient) -> None:
        self.client = client

    @staticmethod
    def should_run(
        *,
        message: str,
        travel_date: date | None,
        include_weather: bool | None,
        required_tools: list[str] | None = None,
    ) -> bool:
        if include_weather is not None:
            return include_weather
        normalized = normalize_text(message)
        return (
            travel_date is not None
            or "weather" in (required_tools or [])
            or any(term in normalized for term in WEATHER_TERMS)
        )

    def run(
        self,
        *,
        message: str,
        city: str | None,
        travel_date: date | None,
        latitude: float | None,
        longitude: float | None,
    ) -> WeatherAssessment:
        location = _city_location(city, message)
        if latitude is not None and longitude is not None:
            location_name = city or "Vị trí đã chọn"
        elif location is not None:
            location_name, latitude, longitude = location
        else:
            raise WeatherLocationRequired(
                "Cần city hoặc latitude/longitude để kiểm tra thời tiết."
            )
        target_date = travel_date or _target_date(message)
        forecast = self.client.forecast(
            latitude,
            longitude,
            target_date,
            vietnam_today(),
        )
        suitability, advice = assess_suitability(
            precipitation=forecast.precipitation_probability,
            thunderstorm=forecast.thunderstorm_probability,
            uv_index=forecast.uv_index,
            wind_gust=forecast.wind_gust_kph,
            max_temperature=forecast.max_temperature_c,
        )
        return WeatherAssessment(
            location=location_name,
            **forecast.__dict__,
            suitability=suitability,
            advice=advice,
        )


def _city_location(city: str | None, message: str) -> tuple[str, float, float] | None:
    haystack = normalize_text(" ".join(part for part in (city, message) if part))
    for key, location in CITY_COORDINATES.items():
        if key in haystack:
            return location
    return None


def _target_date(message: str) -> date:
    normalized = normalize_text(message)
    if "ngay mai" in normalized:
        return vietnam_today() + timedelta(days=1)
    iso_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
    if iso_date:
        try:
            return date.fromisoformat(iso_date.group(1))
        except ValueError:
            pass
    return vietnam_today()


def assess_suitability(
    *,
    precipitation: float | None,
    thunderstorm: float | None,
    uv_index: float | None,
    wind_gust: float | None,
    max_temperature: float | None,
) -> tuple[str, str]:
    if (thunderstorm or 0) >= 40 or (precipitation or 0) >= 70 or (wind_gust or 0) >= 50:
        return (
            "unsuitable",
            "Không thuận lợi cho hoạt động ngoài trời; nên ưu tiên điểm trong nhà hoặc đổi thời gian.",
        )
    if (
        (precipitation or 0) >= 40
        or (uv_index or 0) >= 8
        or (wind_gust or 0) >= 35
        or (max_temperature or 0) >= 36
    ):
        return (
            "caution",
            "Có thể đi nhưng nên chuẩn bị phương án dự phòng và kiểm tra lại dự báo trước khi khởi hành.",
        )
    return "suitable", "Thời tiết nhìn chung phù hợp để tham quan và vui chơi."
