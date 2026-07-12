from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.apis.routers import include_api_routers
from src.config import settings
from src.infra.kb_client import KbClient
from src.infra.llm import create_answer_generator
from src.shared.logging import configure_logging, install_request_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_settings = settings()
    kb_client = KbClient(app_settings.nextrip_kb_base_url)
    answer_generator = create_answer_generator(app_settings)
    app.state.kb_client = kb_client
    app.state.answer_generator = answer_generator
    try:
        yield
    finally:
        kb_client.close()
        if answer_generator is not None:
            answer_generator.close()


def create_app() -> FastAPI:
    app_settings = settings()
    configure_logging(service="nextrip-be", level=app_settings.log_level)
    app = FastAPI(title="NexTripAI Backend", lifespan=lifespan)
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
    uvicorn.run("src.app:app", host=app_settings.nextrip_be_host, port=app_settings.nextrip_be_port, reload=True)


if __name__ == "__main__":
    main()
