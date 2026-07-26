from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.apis.domains.chat.worker_pool import ChatWorkerPool
from src.apis.domains.chat.idempotency import IdempotencyCoordinator
from src.apis.routers import include_api_routers
from src.config import settings
from src.infra.kb_client import KbClient
from src.infra.chat_store import create_chat_store
from src.infra.weather import OpenMeteoWeatherClient
from src.infra.llm import (
    create_answer_generator,
    create_conversation_contextualizer,
)
from src.shared.logging import configure_logging, install_request_logging
from src.shared.telemetry import configure_telemetry
from src.security.auth import Authenticator
from src.security.rate_limit import InMemoryRateLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_settings = settings()
    kb_client = KbClient(
        app_settings.nextrip_kb_base_url,
        timeout_seconds=app_settings.kb_request_timeout_seconds,
        auth_mode=app_settings.kb_auth_mode,
        auth_audience=app_settings.kb_auth_audience,
        retry_attempts=app_settings.resilience_retry_attempts,
        circuit_failure_threshold=app_settings.circuit_breaker_failure_threshold,
        circuit_recovery_seconds=app_settings.circuit_breaker_recovery_seconds,
    )
    answer_generator = create_answer_generator(app_settings)
    conversation_contextualizer = create_conversation_contextualizer(app_settings)
    weather_client = OpenMeteoWeatherClient(
        app_settings.weather_timeout_seconds,
        retry_attempts=app_settings.resilience_retry_attempts,
        circuit_failure_threshold=app_settings.circuit_breaker_failure_threshold,
        circuit_recovery_seconds=app_settings.circuit_breaker_recovery_seconds,
    )
    chat_store = create_chat_store(app_settings)
    chat_worker_pool = ChatWorkerPool(
        worker_count=app_settings.ai_worker_count,
        queue_capacity=app_settings.ai_queue_capacity,
        weather_client=weather_client,
        chat_store=chat_store,
        chat_history_limit=app_settings.chat_history_limit,
        conversation_contextualizer=conversation_contextualizer,
    )
    app.state.kb_client = kb_client
    app.state.answer_generator = answer_generator
    app.state.conversation_contextualizer = conversation_contextualizer
    app.state.weather_client = weather_client
    app.state.chat_store = chat_store
    app.state.settings = app_settings
    app.state.authenticator = Authenticator(app_settings)
    app.state.rate_limiter = InMemoryRateLimiter(
        app_settings.rate_limit_requests,
        app_settings.rate_limit_window_seconds,
    )
    app.state.idempotency = IdempotencyCoordinator()
    try:
        async with chat_worker_pool.run():
            app.state.chat_worker_pool = chat_worker_pool
            yield
    finally:
        kb_client.close()
        if answer_generator is not None:
            answer_generator.close()
        if conversation_contextualizer is not None:
            conversation_contextualizer.close()
        weather_client.close()
        chat_store.close()


def create_app() -> FastAPI:
    app_settings = settings()
    configure_logging(
        service="nextrip-be",
        level=app_settings.log_level,
        serialize=app_settings.log_json,
    )
    app = FastAPI(title="NexTripAI Backend", lifespan=lifespan)
    configure_telemetry(app, app_settings)
    install_request_logging(app)
    if app_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    include_api_routers(app)
    return app


app = create_app()


def main() -> None:
    app_settings = settings()
    uvicorn.run(
        "src.app:app",
        host=app_settings.nextrip_be_host,
        port=app_settings.nextrip_be_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
