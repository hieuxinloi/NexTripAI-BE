from __future__ import annotations

import re
from typing import TypeAlias


KB_VERSION_PATTERN = re.compile(r"^v[1-9][0-9]*$")
KbVersion: TypeAlias = str

DEFAULT_KB_VERSION: KbVersion = "v1"
DEFAULT_TYPED_KB_VERSION: KbVersion = "v2"
STRUCTURED_CONVERSATION_CONTEXT_MIN_VERSION = 8


def is_kb_version(value: str) -> bool:
    return bool(KB_VERSION_PATTERN.fullmatch(value))


def is_typed_kb_version(value: str | None) -> bool:
    return bool(value and value != "v1" and is_kb_version(value))


def supports_structured_conversation_context(value: str) -> bool:
    return (
        is_kb_version(value)
        and int(value.removeprefix("v"))
        >= STRUCTURED_CONVERSATION_CONTEXT_MIN_VERSION
    )
