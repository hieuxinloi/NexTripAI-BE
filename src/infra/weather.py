from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx


WEATHER_ENDPOINT = "https://weather.googleapis.com/v1/forecast/days:lookup"


class WeatherUnavailable(RuntimeError):
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


class GoogleWeatherClient:
    def __init__(
        self,
        api_key: str | None,
        timeout_seconds: float = 8.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds, trust_env=False)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def forecast(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
        today: date,
    ) -> DailyForecast:
        if not self.api_key:
            raise WeatherUnavailable("Google Weather API key is not configured.")
        days_ahead = (target_date - today).days
        if days_ahead < 0 or days_ahead > 9:
            raise WeatherUnavailable("Weather forecast supports today through the next 9 days.")
        try:
            response = self.client.get(
                WEATHER_ENDPOINT,
                params={
                    "key": self.api_key,
                    "location.latitude": latitude,
                    "location.longitude": longitude,
                    "days": days_ahead + 1,
                    "pageSize": 10,
                    "languageCode": "vi",
                    "unitsSystem": "METRIC",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WeatherUnavailable("Google Weather API request failed.") from exc
        forecast = _select_day(response.json().get("forecastDays", []), target_date)
        if forecast is None:
            raise WeatherUnavailable("No forecast is available for the requested date.")
        return _daily_forecast(target_date, forecast)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def _select_day(rows: list[dict[str, Any]], target: date) -> dict[str, Any] | None:
    for row in rows:
        display = row.get("displayDate") or {}
        if (display.get("year"), display.get("month"), display.get("day")) == (
            target.year,
            target.month,
            target.day,
        ):
            return row
    return None


def _number(data: dict[str, Any], *path: str) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _daily_forecast(target_date: date, forecast: dict[str, Any]) -> DailyForecast:
    daytime = forecast.get("daytimeForecast") or {}
    condition = daytime.get("weatherCondition") or {}
    description = condition.get("description") or {}
    precipitation = _number(daytime, "precipitation", "probability", "percent")
    thunderstorm = _number(daytime, "thunderstormProbability")
    uv_index = _number(daytime, "uvIndex")
    return DailyForecast(
        forecast_date=target_date,
        condition=description.get("text") or condition.get("type") or "Không rõ",
        condition_type=condition.get("type"),
        min_temperature_c=_number(forecast, "minTemperature", "degrees"),
        max_temperature_c=_number(forecast, "maxTemperature", "degrees"),
        precipitation_probability=int(precipitation) if precipitation is not None else None,
        thunderstorm_probability=int(thunderstorm) if thunderstorm is not None else None,
        uv_index=int(uv_index) if uv_index is not None else None,
        wind_gust_kph=_number(daytime, "wind", "gust", "value"),
    )
