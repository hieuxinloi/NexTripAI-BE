from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    nextrip_be_host: str = "0.0.0.0"
    nextrip_be_port: int = 8000
    nextrip_kb_base_url: str = "http://127.0.0.1:8010"
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"
    ai_worker_count: int = Field(default=5, ge=1, le=32)
    ai_queue_capacity: int = Field(default=100, ge=1, le=10_000)
    answer_generation_mode: str = "template"
    gemini_model: str = "gemini-2.5-flash"
    answer_temperature: float = 0.2
    google_api_key: str | None = None
    google_genai_use_vertexai: bool = False
    google_application_credentials: str | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_allowed_origins.split(",") if item.strip()]


@lru_cache
def settings() -> Settings:
    return Settings()
