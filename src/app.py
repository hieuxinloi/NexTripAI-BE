from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.apis.routers import include_api_routers
from src.config import settings
from src.shared.logging import configure_logging, install_request_logging


def create_app() -> FastAPI:
    app_settings = settings()
    configure_logging(service="nextrip-be", level=app_settings.log_level)
    app = FastAPI(title="NexTripAI Backend")
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
