from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.infra.chat_store import InMemoryChatStore, create_chat_store


def test_memory_store_keeps_recent_messages_in_order() -> None:
    store = InMemoryChatStore()
    store.save_message("session", "1", "user", "one")
    store.save_message("session", "2", "assistant", "two")

    assert [item["content"] for item in store.recent_messages("session", 2)] == [
        "one",
        "two",
    ]


def test_chat_store_defaults_to_memory() -> None:
    app_settings = Settings(
        _env_file=None,
        gemini_context_model="test-context-model",
        gemini_answer_model="test-answer-model",
        gemini_thinking_level="minimal",
    )
    assert create_chat_store(app_settings).backend_name == "memory"


def test_kb_fallback_defaults_to_disabled() -> None:
    app_settings = Settings(
        _env_file=None,
        gemini_context_model="test-context-model",
        gemini_answer_model="test-answer-model",
        gemini_thinking_level="minimal",
    )
    assert app_settings.kb_fallback_version_list == []


def test_gemini_models_are_required_from_environment(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_CONTEXT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_ANSWER_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)

    with pytest.raises(ValidationError, match="gemini_context_model"):
        Settings(_env_file=None)


def test_memory_store_enforces_session_owner_and_deletes_session() -> None:
    store = InMemoryChatStore()
    store.save_message("session-1", "m1", "user", "hello", user_id="user-a")

    assert store.delete_session("session-1", user_id="user-a") is True
    assert store.recent_messages("session-1", 10, user_id="user-a") == []


def test_memory_store_rejects_another_session_owner() -> None:
    store = InMemoryChatStore()
    store.save_message("session-1", "m1", "user", "hello", user_id="user-a")

    try:
        store.recent_messages("session-1", 10, user_id="user-b")
    except PermissionError:
        pass
    else:
        raise AssertionError("Expected session ownership check")


def test_memory_store_round_trips_idempotent_response() -> None:
    store = InMemoryChatStore()
    store.save_idempotent_response(
        "session-1",
        "request-key",
        {"answer": "cached"},
        user_id="user-a",
        ttl_seconds=60,
    )

    assert store.get_idempotent_response(
        "session-1",
        "request-key",
        user_id="user-a",
    ) == {"answer": "cached"}


def test_memory_store_round_trips_session_memory() -> None:
    store = InMemoryChatStore()
    store.save_session_memory(
        "session-1",
        {"summary": "User prefers beaches."},
        user_id="user-a",
    )

    assert store.get_session_memory(
        "session-1",
        user_id="user-a",
    ) == {"summary": "User prefers beaches."}

def test_memory_store_lists_sessions_without_deleting_previous_chat() -> None:
    store = InMemoryChatStore()
    store.create_session("older", user_id="user-a", title="Cà phê Quy Nhơn")
    store.save_message("newer", "m1", "user", "Nhà hàng ở Đà Nẵng", user_id="user-a")

    sessions = store.list_sessions(user_id="user-a")

    assert [item["session_id"] for item in sessions] == ["newer", "older"]
    assert sessions[0]["title"] == "Nhà hàng ở Đà Nẵng"
    assert sessions[0]["message_count"] == 1


def test_memory_store_renames_owned_session() -> None:
    store = InMemoryChatStore()
    store.create_session("session-1", user_id="user-a", title="Tên cũ")

    renamed = store.rename_session(
        "session-1",
        "Lịch trình Quy Nhơn",
        user_id="user-a",
    )

    assert renamed is not None
    assert renamed["title"] == "Lịch trình Quy Nhơn"
    assert store.list_sessions(user_id="user-a")[0]["title"] == "Lịch trình Quy Nhơn"
