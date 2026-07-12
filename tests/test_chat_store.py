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
