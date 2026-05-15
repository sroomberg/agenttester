"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str: ...
