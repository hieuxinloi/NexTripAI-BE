from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.apis.domains.chat.worker_pool import ChatWorkerPool
from src.apis.domains.chat.idempotency import IdempotencyCoordinator
from src.apis.domains.evaluations.manager import EvaluationManager
from src.apis.routers import include_api_routers
from src.config import settings
from src.infra.kb_client import KbClient
from src.infra.current_data_client import CurrentDataClient
from src.infra.chat_store import create_chat_store
from src.infra.saved_trip_store import create_saved_trip_store
from src.infra.user_profile_store import create_user_profile_store
from src.infra.weather import OpenMeteoWeatherClient
from src.infra.llm import (
    create_answer_generator,
    create_conversation_contextualizer,
)
from src.infra.evaluation_judge import create_evaluation_judge
from src.infra.evaluation_store import create_evaluation_store
from src.shared.logging import configure_logging, install_request_logging
from src.shared.telemetry import configure_telemetry
from src.security.auth import Authenticator
from src.security.rate_limit import InMemoryRateLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_settings = settings()
    kb_client = KbClient(
        app_settings.nextrip_kb_base_url,
        timeout_seconds=app_settings.effective_kb_request_timeout_seconds,
        auth_mode=app_settings.kb_auth_mode,
        auth_audience=app_settings.kb_auth_audience,
        retry_attempts=app_settings.resilience_retry_attempts,
        circuit_failure_threshold=app_settings.circuit_breaker_failure_threshold,
        circuit_recovery_seconds=app_settings.circuit_breaker_recovery_seconds,
        admin_api_key=app_settings.kb_admin_api_key,
    )
    answer_generator = create_answer_generator(app_settings)
    conversation_contextualizer = create_conversation_contextualizer(app_settings)
    evaluation_judge = create_evaluation_judge(app_settings)
    evaluation_store = create_evaluation_store(app_settings)
    weather_client = OpenMeteoWeatherClient(
        app_settings.weather_timeout_seconds,
        retry_attempts=app_settings.resilience_retry_attempts,
        circuit_failure_threshold=app_settings.circuit_breaker_failure_threshold,
        circuit_recovery_seconds=app_settings.circuit_breaker_recovery_seconds,
    )
    current_data_client = None
    if app_settings.current_data_enabled:
        if not app_settings.current_data_api_key:
            raise RuntimeError(
                "CURRENT_DATA_API_KEY is required when CURRENT_DATA_ENABLED=true"
            )
        current_data_client = CurrentDataClient(
            app_settings.current_data_base_url,
            app_settings.current_data_api_key,
            timeout_seconds=app_settings.current_data_timeout_seconds,
            retry_attempts=app_settings.resilience_retry_attempts,
            circuit_failure_threshold=app_settings.circuit_breaker_failure_threshold,
            circuit_recovery_seconds=app_settings.circuit_breaker_recovery_seconds,
        )
    chat_store = create_chat_store(app_settings)
    saved_trip_store = create_saved_trip_store(app_settings)
    user_profile_store = create_user_profile_store(app_settings)
    chat_worker_pool = ChatWorkerPool(
        worker_count=app_settings.ai_worker_count,
        queue_capacity=app_settings.ai_queue_capacity,
        weather_client=weather_client,
        chat_store=chat_store,
        chat_history_limit=app_settings.chat_history_limit,
        conversation_contextualizer=conversation_contextualizer,
        user_profile_store=user_profile_store,
        current_data_client=current_data_client,
    )
    evaluation_manager = EvaluationManager(
        worker_pool=chat_worker_pool,
        kb_client=kb_client,
        answer_generator=answer_generator,
        chat_store=chat_store,
        evaluation_store=evaluation_store,
        judge=evaluation_judge,
        chat_timeout_seconds=app_settings.chat_request_timeout_seconds,
    )
    app.state.kb_client = kb_client
    app.state.answer_generator = answer_generator
    app.state.conversation_contextualizer = conversation_contextualizer
    app.state.evaluation_manager = evaluation_manager
    app.state.weather_client = weather_client
    app.state.current_data_client = current_data_client
    app.state.chat_store = chat_store
    app.state.saved_trip_store = saved_trip_store
    app.state.user_profile_store = user_profile_store
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
        await evaluation_manager.close()
        kb_client.close()
        if answer_generator is not None:
            answer_generator.close()
        if conversation_contextualizer is not None:
            conversation_contextualizer.close()
        weather_client.close()
        if current_data_client is not None:
            current_data_client.close()
        chat_store.close()
        saved_trip_store.close()
        user_profile_store.close()


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
