"""HTTP client for vLLM OpenAI-compatible inference servers."""

from __future__ import annotations

import aiohttp


async def check_connection(endpoint: str, timeout: int = 5) -> bool:
    """Return True if the vLLM server at *endpoint* is reachable."""
    try:
        _timeout = aiohttp.ClientTimeout(total=timeout)
        async with (
            aiohttp.ClientSession(timeout=_timeout) as session,
            session.get(f"{endpoint.rstrip('/')}/v1/models") as resp,
        ):
            return resp.status < 500
    except Exception:
        return False
