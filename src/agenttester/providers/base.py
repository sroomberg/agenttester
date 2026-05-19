"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import aiohttp

# Shared timeouts: 30s connect, 300s read for non-streaming calls,
# no read timeout for streaming (model may take arbitrarily long).
_API_CALL_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_read=300)
_API_STREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_read=None)


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str: ...
