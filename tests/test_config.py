from __future__ import annotations

import pytest

from src.config import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        gemini_context_model="context-model",
        gemini_answer_model="answer-model",
        gemini_thinking_level="minimal",
        **overrides,
    )


def test_kb_timeout_is_derived_from_chat_budget() -> None:
    settings = _settings(
        chat_request_timeout_seconds=80,
        kb_request_timeout_ratio=0.625,
    )

    assert settings.effective_kb_request_timeout_seconds == pytest.approx(50)


def test_explicit_kb_timeout_cannot_consume_the_entire_chat_budget() -> None:
    settings = _settings(
        chat_request_timeout_seconds=80,
        kb_request_timeout_ratio=0.625,
        kb_request_timeout_seconds=70,
    )

    assert settings.effective_kb_request_timeout_seconds == pytest.approx(50)
