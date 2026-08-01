from __future__ import annotations

import os


# Automated tests intentionally exercise the non-Firebase code paths without
# contacting a real identity provider. Disabled auth is fail-closed to `user`.
os.environ["AUTH_MODE"] = "disabled"
os.environ["CHAT_STORE_BACKEND"] = "memory"
os.environ["ANSWER_GENERATION_MODE"] = "template"
os.environ["CONVERSATION_CONTEXT_ENABLED"] = "false"
os.environ["GOOGLE_API_KEY"] = ""
