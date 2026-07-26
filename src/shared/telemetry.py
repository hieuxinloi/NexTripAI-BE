from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from fastapi import FastAPI
from loguru import logger

from src.config import Settings


_tracer = None
_llm_tokens = None
_llm_cost = None


def configure_telemetry(app: FastAPI, app_settings: Settings) -> None:
    global _tracer, _llm_tokens, _llm_cost
    if not app_settings.telemetry_enabled:
        return
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError(
            "TELEMETRY_ENABLED=true requires the OpenTelemetry dependencies."
        ) from exc

    resource = Resource.create({"service.name": app_settings.otel_service_name})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(app_settings.otel_service_name)

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter(app_settings.otel_service_name)
    _llm_tokens = meter.create_counter(
        "nextrip.llm.tokens",
        description="Gemini token usage by direction and model",
        unit="{token}",
    )
    _llm_cost = meter.create_counter(
        "nextrip.llm.estimated_cost",
        description="Estimated Gemini cost using configured per-million-token rates",
        unit="USD",
    )
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry enabled service={}", app_settings.otel_service_name)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name, attributes=attributes):
        yield


def record_llm_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    thinking_tokens: int = 0,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> None:
    if _llm_tokens is None:
        return
    _llm_tokens.add(input_tokens, {"model": model, "direction": "input"})
    _llm_tokens.add(output_tokens, {"model": model, "direction": "output"})
    _llm_tokens.add(thinking_tokens, {"model": model, "direction": "thinking"})
    if _llm_cost is not None:
        estimated_cost = (
            input_tokens * input_cost_per_million
            + (output_tokens + thinking_tokens) * output_cost_per_million
        ) / 1_000_000
        _llm_cost.add(estimated_cost, {"model": model})
