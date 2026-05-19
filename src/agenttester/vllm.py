"""HTTP client for vLLM OpenAI-compatible inference servers."""

from __future__ import annotations

import aiohttp

from .providers.base import _API_CALL_TIMEOUT


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


async def query(
    endpoint: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int = 2048,
    api_key: str | None = None,
) -> str:
    """Send a chat completion request and return the response text."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
    async with (
        aiohttp.ClientSession(timeout=_API_CALL_TIMEOUT) as session,
        session.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json=body,
            headers=headers,
        ) as resp,
    ):
        data = await resp.json()
    return data["choices"][0]["message"]["content"]
