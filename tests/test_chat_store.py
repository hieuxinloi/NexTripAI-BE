from __future__ import annotations

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
    assert create_chat_store(Settings()).backend_name == "memory"


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
