"""Persistent REPL session management."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _default_sessions_dir() -> Path:
    return Path.home() / ".config" / "agenttester" / "sessions"


@dataclass
class ReplSession:
    """Persisted REPL session: one message history list per model, keyed by name."""

    id: str
    created_at: str
    histories: dict[str, list[dict]] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str) -> ReplSession:
        return cls(
            id=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, name: str, sessions_dir: Path | None = None) -> ReplSession:
        path = (sessions_dir or _default_sessions_dir()) / f"{name}.json"
        data = json.loads(path.read_text())
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            histories=data.get("histories", {}),
        )

    @classmethod
    def load_or_create(
        cls, name: str, sessions_dir: Path | None = None
    ) -> tuple[ReplSession, bool]:
        """Load an existing session or create a new one.

        Returns ``(session, is_new)`` — ``is_new`` is True when a fresh
        session was created.
        """
        try:
            return cls.load(name, sessions_dir), False
        except (FileNotFoundError, KeyError):
            return cls.create(name), True

    def save(self, sessions_dir: Path | None = None) -> None:
        d = sessions_dir or _default_sessions_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": self.id,
                    "created_at": self.created_at,
                    "histories": self.histories,
                },
                indent=2,
            )
        )

    def delete(self, sessions_dir: Path | None = None) -> None:
        path = (sessions_dir or _default_sessions_dir()) / f"{self.id}.json"
        path.unlink(missing_ok=True)

    @classmethod
    def list_all(cls, sessions_dir: Path | None = None) -> list[ReplSession]:
        d = sessions_dir or _default_sessions_dir()
        if not d.exists():
            return []
        sessions = []
        for p in sorted(d.glob("*.json")):
            with contextlib.suppress(Exception):
                sessions.append(cls.load(p.stem, d))
        return sessions
