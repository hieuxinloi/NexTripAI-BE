from __future__ import annotations

from typing import Literal, TypeAlias


KbVersion: TypeAlias = Literal["v1", "v2", "v3", "v4", "v5"]

DEFAULT_KB_VERSION: KbVersion = "v1"
DEFAULT_TYPED_KB_VERSION: KbVersion = "v2"
TYPED_KB_VERSIONS = frozenset({"v2", "v3", "v4", "v5"})
