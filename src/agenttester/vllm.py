"""HTTP client for vLLM OpenAI-compatible inference servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def check_connection(endpoint: str, timeout: int = 5) -> bool:
    """Return True if the vLLM server at *endpoint* is reachable.

    Uses GET /v1/models — a lightweight, read-only health check that every
    vLLM server exposes.
    """
    try:
        req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/models")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def query(
    endpoint: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int = 2048,
    timeout: int = 120,
) -> str:
    """Send a chat completion request and return the response text.

    Raises urllib.error.HTTPError or OSError on failure.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]
