from __future__ import annotations

from functools import lru_cache
import re
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_KB_VERSION_PATTERN = re.compile(r"^v[1-9][0-9]*$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    nextrip_be_host: str = "0.0.0.0"
    nextrip_be_port: int = 8000
    nextrip_kb_base_url: str = "http://127.0.0.1:8010"
    environment: Literal["local", "test", "production"] = "local"
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"
    log_json: bool = False
    ai_worker_count: int = Field(default=5, ge=1, le=32)
    ai_queue_capacity: int = Field(default=100, ge=1, le=10_000)
    answer_generation_mode: str = "template"
    gemini_context_model: str = Field(min_length=1)
    gemini_answer_model: str = Field(min_length=1)
    gemini_planning_model: str = "gemini-3.5-flash-lite"
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"]
    gemini_planning_thinking_level: Literal["minimal", "low", "medium", "high"] = "medium"
    answer_temperature: float = 0.2
    gemini_timeout_seconds: float = Field(default=35.0, gt=0, le=60)
    gemini_input_cost_per_million_usd: float = Field(default=0.0, ge=0)
    gemini_output_cost_per_million_usd: float = Field(default=0.0, ge=0)
    google_api_key: str | None = None
    google_cloud_project: str | None = None
    conversation_context_enabled: bool = True
    conversation_context_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    conversation_summary_max_chars: int = Field(default=1200, ge=200, le=4000)
    weather_timeout_seconds: float = Field(default=8.0, gt=0, le=30)
    chat_store_backend: str = "memory"
    firestore_credentials_path: str | None = None
    firestore_database: str = "(default)"
    firestore_sessions_collection: str = "chat_sessions"
    chat_history_limit: int = Field(default=8, ge=1, le=50)
    auth_mode: Literal["disabled", "firebase"] = "disabled"
    firebase_project_id: str | None = None
    kb_auth_mode: Literal["none", "google_oidc"] = "none"
    kb_auth_audience: str | None = None
    active_kb_version: str = Field(default="v8", pattern=r"^v[1-9][0-9]*$")
    kb_fallback_versions: str = ""
    allow_client_kb_version: bool = True
    kb_request_timeout_seconds: float = Field(default=25.0, gt=0, le=60)
    chat_request_timeout_seconds: float = Field(default=45.0, gt=1, le=120)
    rate_limit_requests: int = Field(default=30, ge=1, le=10_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3_600)
    idempotency_ttl_seconds: int = Field(default=900, ge=60, le=86_400)
    resilience_retry_attempts: int = Field(default=2, ge=1, le=5)
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1, le=100)
    circuit_breaker_recovery_seconds: float = Field(default=30.0, gt=0, le=600)
    telemetry_enabled: bool = False
    otel_service_name: str = "nextrip-be"

    @property
    def cors_origins(self) -> list[str]:
        return [
            item.strip()
            for item in self.cors_allowed_origins.split(",")
            if item.strip()
        ]

    @property
    def kb_fallback_version_list(self) -> list[str]:
        versions = [
            item.strip().lower()
            for item in self.kb_fallback_versions.split(",")
            if item.strip()
        ]
        return [item for item in versions if _KB_VERSION_PATTERN.fullmatch(item)]


@lru_cache
def settings() -> Settings:
    return Settings()
