"""Shared state for models waiting on user input."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Sentinel value indicating the wait was interrupted (timeout or exit),
# distinct from any real user response.
WAIT_INTERRUPTED = object()

# Default wait timeout: 30 minutes
DEFAULT_TIMEOUT = 1800.0


@dataclass
class PendingQuestion:
    model_name: str
    question: str
    asked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response: str | object | None = None
    answered: threading.Event = field(default_factory=threading.Event)


class QuestionRegistry:
    """Thread-safe registry of models waiting for user input."""

    def __init__(self, default_timeout: float = DEFAULT_TIMEOUT) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, PendingQuestion] = {}
        self._default_timeout = default_timeout

    def ask(
        self, model_name: str, question: str, timeout: float | None = None
    ) -> str | None:
        """Block until the user responds or timeout expires.

        Returns the user's response string, or None if the wait was
        interrupted (timeout or session exit). Callers should treat None
        as "save state and stop cleanly" rather than injecting a message.
        """
        entry = PendingQuestion(model_name=model_name, question=question)
        with self._lock:
            self._pending[model_name] = entry

        effective_timeout = timeout if timeout is not None else self._default_timeout
        entry.answered.wait(timeout=effective_timeout)

        with self._lock:
            self._pending.pop(model_name, None)

        if not entry.answered.is_set() or entry.response is WAIT_INTERRUPTED:
            return None
        return entry.response  # type: ignore[return-value]

    def respond(self, model_name: str, response: str) -> bool:
        """Deliver a user response to a waiting model. Returns False if not waiting."""
        with self._lock:
            entry = self._pending.get(model_name)
            if entry is None:
                return False
        entry.response = response
        entry.answered.set()
        return True

    def pending(self) -> list[PendingQuestion]:
        """Return a snapshot of all pending questions."""
        with self._lock:
            return list(self._pending.values())

    def cancel_all(self) -> None:
        """Unblock all waiting models without injecting messages."""
        with self._lock:
            entries = list(self._pending.values())
        for entry in entries:
            if not entry.answered.is_set():
                entry.response = WAIT_INTERRUPTED
                entry.answered.set()
