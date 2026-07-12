from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx


WEATHER_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
DAILY_VARIABLES = ",".join((
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "uv_index_max",
    "wind_gusts_10m_max",
))
WMO_CONDITIONS = {
    0: ("Trời quang", "CLEAR_SKY"),
    1: ("Trời chủ yếu quang", "MAINLY_CLEAR"),
    2: ("Có mây rải rác", "PARTLY_CLOUDY"),
    3: ("Nhiều mây", "OVERCAST"),
    45: ("Có sương mù", "FOG"),
    48: ("Sương mù đóng băng", "RIME_FOG"),
    51: ("Mưa phùn nhẹ", "LIGHT_DRIZZLE"),
    53: ("Mưa phùn", "DRIZZLE"),
    55: ("Mưa phùn dày", "DENSE_DRIZZLE"),
    56: ("Mưa phùn đóng băng nhẹ", "LIGHT_FREEZING_DRIZZLE"),
    57: ("Mưa phùn đóng băng", "FREEZING_DRIZZLE"),
    61: ("Mưa nhẹ", "LIGHT_RAIN"),
    63: ("Mưa vừa", "MODERATE_RAIN"),
    65: ("Mưa lớn", "HEAVY_RAIN"),
    66: ("Mưa đóng băng nhẹ", "LIGHT_FREEZING_RAIN"),
    67: ("Mưa đóng băng lớn", "HEAVY_FREEZING_RAIN"),
    71: ("Tuyết rơi nhẹ", "LIGHT_SNOW"),
    73: ("Tuyết rơi vừa", "MODERATE_SNOW"),
    75: ("Tuyết rơi dày", "HEAVY_SNOW"),
    77: ("Hạt tuyết", "SNOW_GRAINS"),
    80: ("Mưa rào nhẹ", "LIGHT_RAIN_SHOWERS"),
    81: ("Mưa rào", "RAIN_SHOWERS"),
    82: ("Mưa rào mạnh", "HEAVY_RAIN_SHOWERS"),
    85: ("Mưa tuyết nhẹ", "LIGHT_SNOW_SHOWERS"),
    86: ("Mưa tuyết mạnh", "HEAVY_SNOW_SHOWERS"),
    95: ("Có giông", "THUNDERSTORM"),
    96: ("Giông kèm mưa đá nhẹ", "THUNDERSTORM_WITH_HAIL"),
    99: ("Giông kèm mưa đá mạnh", "HEAVY_THUNDERSTORM_WITH_HAIL"),
}


class WeatherUnavailable(RuntimeError):
    pass


class WeatherLocationRequired(WeatherUnavailable):
    pass


@dataclass(frozen=True)
class DailyForecast:
    forecast_date: date
    condition: str
    condition_type: str | None
    min_temperature_c: float | None
    max_temperature_c: float | None
    precipitation_probability: int | None
    thunderstorm_probability: int | None
    uv_index: int | None
    wind_gust_kph: float | None


class OpenMeteoWeatherClient:
    def __init__(
        self,
        timeout_seconds: float = 8.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds, trust_env=False)

    @property
    def configured(self) -> bool:
        return True

    def forecast(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
        today: date,
    ) -> DailyForecast:
        days_ahead = (target_date - today).days
        if days_ahead < 0 or days_ahead > 15:
            raise WeatherUnavailable(
                "Open-Meteo forecast supports today through the next 15 days."
            )
        try:
            response = self.client.get(
                WEATHER_ENDPOINT,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": DAILY_VARIABLES,
                    "timezone": "Asia/Ho_Chi_Minh",
                    "start_date": target_date.isoformat(),
                    "end_date": target_date.isoformat(),
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WeatherUnavailable("Open-Meteo weather request failed.") from exc
        return _daily_forecast(target_date, response.json().get("daily") or {})

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def _daily_forecast(target_date: date, daily: dict[str, Any]) -> DailyForecast:
    times = daily.get("time")
    if not isinstance(times, list) or target_date.isoformat() not in times:
        raise WeatherUnavailable("No Open-Meteo forecast is available for the requested date.")
    index = times.index(target_date.isoformat())
    weather_code = _integer_value(daily, "weather_code", index)
    condition, condition_type = WMO_CONDITIONS.get(
        weather_code,
        ("Không rõ", f"WMO_{weather_code}" if weather_code is not None else None),
    )
    return DailyForecast(
        forecast_date=target_date,
        condition=condition,
        condition_type=condition_type,
        min_temperature_c=_number_value(daily, "temperature_2m_min", index),
        max_temperature_c=_number_value(daily, "temperature_2m_max", index),
        precipitation_probability=_integer_value(
            daily,
            "precipitation_probability_max",
            index,
        ),
        thunderstorm_probability=100 if weather_code in {95, 96, 99} else None,
        uv_index=_integer_value(daily, "uv_index_max", index),
        wind_gust_kph=_number_value(daily, "wind_gusts_10m_max", index),
    )


def _number_value(data: dict[str, Any], key: str, index: int) -> float | None:
    values = data.get(key)
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    return float(value) if isinstance(value, (int, float)) else None


def _integer_value(data: dict[str, Any], key: str, index: int) -> int | None:
    value = _number_value(data, key, index)
    return round(value) if value is not None else None
