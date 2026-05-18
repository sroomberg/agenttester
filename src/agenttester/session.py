"""Persistent REPL session management."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

_DEFAULT_MAX_SESSIONS = 5


def _default_sessions_dir() -> Path:
    return Path.home() / ".config" / "agenttester" / "sessions"


def _read_max_sessions() -> int:
    """Read max_sessions from global config, falling back to the default."""
    config_candidates = [
        Path.home() / ".config" / "agenttester" / "config.yml",
        Path.home() / ".config" / "agenttester" / "config.yaml",
    ]
    for path in config_candidates:
        if path.exists():
            with contextlib.suppress(Exception):
                data = yaml.safe_load(path.read_text()) or {}
                val = data.get("max_sessions")
                if isinstance(val, int) and val > 0:
                    return val
    return _DEFAULT_MAX_SESSIONS


@dataclass
class ReplSession:
    """Persisted REPL session: one message history list per model, keyed by name."""

    id: str
    created_at: str
    histories: dict[str, list[dict]] = field(default_factory=dict)
    branches: list[str] = field(default_factory=list)

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
            branches=data.get("branches", []),
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

    def save(
        self, sessions_dir: Path | None = None, max_sessions: int | None = None
    ) -> None:
        d = sessions_dir or _default_sessions_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": self.id,
                    "created_at": self.created_at,
                    "histories": self.histories,
                    "branches": self.branches,
                },
                indent=2,
            )
        )
        limit = max_sessions if max_sessions is not None else _read_max_sessions()
        self._prune_old(d, limit)

    @classmethod
    def _prune_old(cls, sessions_dir: Path, max_sessions: int) -> None:
        """Delete the oldest sessions beyond max_sessions."""
        all_sessions = cls.list_all(sessions_dir)
        if len(all_sessions) <= max_sessions:
            return
        sorted_sessions = sorted(all_sessions, key=lambda s: s.created_at)
        for s in sorted_sessions[: len(sorted_sessions) - max_sessions]:
            s.delete(sessions_dir)

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
