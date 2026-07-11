from __future__ import annotations

import logging
import sys
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from loguru import logger

from src.shared.request_context import reset_request_id, set_request_id


LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | {extra[service]} | "
    "request_id={extra[request_id]} | <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(*, service: str, level: str = "INFO") -> None:
    logger.remove()
    logger.configure(extra={"service": service, "request_id": "-"})
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=LOG_FORMAT,
        colorize=sys.stderr.isatty(),
        backtrace=False,
        diagnose=False,
    )
    handler = InterceptHandler()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [handler]
        stdlib_logger.propagate = False


def safe_text(value: str | None, max_length: int = 240) -> str:
    if not value:
        return "-"
    compact = " ".join(value.split())
    return compact if len(compact) <= max_length else compact[: max_length - 3] + "..."


def install_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = safe_text(request.headers.get("X-Request-ID"), 80)
        if request_id == "-":
            request_id = uuid4().hex[:16]
        token = set_request_id(request_id)
        started_at = perf_counter()
        try:
            with logger.contextualize(request_id=request_id):
                logger.info(
                    "HTTP request start method={} path={} client={}",
                    request.method,
                    request.url.path,
                    request.client.host if request.client else "-",
                )
                try:
                    response = await call_next(request)
                except Exception as exc:
                    logger.exception(
                        "HTTP request error method={} path={} error_type={} elapsed_ms={}",
                        request.method,
                        request.url.path,
                        exc.__class__.__name__,
                        int((perf_counter() - started_at) * 1000),
                    )
                    raise
                response.headers["X-Request-ID"] = request_id
                logger.info(
                    "HTTP request end method={} path={} status={} elapsed_ms={}",
                    request.method,
                    request.url.path,
                    response.status_code,
                    int((perf_counter() - started_at) * 1000),
                )
                return response
        finally:
            reset_request_id(token)
