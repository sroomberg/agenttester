"""Per-model event logging for session watchers."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import GLOBAL_CONFIG_DIR


def _sessions_dir() -> Path:
    return GLOBAL_CONFIG_DIR / "sessions"


class EventLogger:
    """Appends structured events to a per-model JSONL file.

    Thread-safe: may be called from asyncio background threads.
    """

    @staticmethod
    def path_for(session_id: str, model_name: str) -> Path:
        safe = model_name.replace("/", "-").replace(" ", "-")
        return _sessions_dir() / session_id / "events" / f"{safe}.jsonl"

    def __init__(self, session_id: str, model_name: str) -> None:
        self._path = self.path_for(session_id, model_name)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event_type: str, content: str) -> None:
        entry = json.dumps(
            {
                "type": event_type,
                "content": content,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        with self._lock, self._path.open("a") as f:
            f.write(entry + "\n")

    @property
    def path(self) -> Path:
        return self._path
