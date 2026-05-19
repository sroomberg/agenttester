"""Abstract base class for LLM providers."""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import aiohttp

# Shared timeouts: 30s connect, 300s read for non-streaming calls,
# no read timeout for streaming (model may take arbitrarily long).
_API_CALL_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_read=300)
_API_STREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_read=None)

_CLI_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_CLI_TOKEN_TTL = 55 * 60  # 55 min — Azure/GCP tokens are valid ~60 min


def _fetch_cli_token(command: str) -> str:
    """Run a shell command to obtain a bearer token, caching for 55 minutes."""
    cached = _CLI_TOKEN_CACHE.get(command)
    if cached and time.time() < cached[1]:
        return cached[0]
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    token = result.stdout.strip()
    if not token:
        raise RuntimeError(
            f"CLI token fetch failed: {result.stderr.strip() or '(no output)'}"
        )
    _CLI_TOKEN_CACHE[command] = (token, time.time() + _CLI_TOKEN_TTL)
    return token


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str: ...

    @abstractmethod
    async def async_stream_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict: ...
