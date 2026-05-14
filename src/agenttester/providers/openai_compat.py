"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
import os
import urllib.request

from .base import Provider


class OpenAICompatProvider(Provider):
    """Calls any OpenAI-compatible chat completions endpoint.

    Covers vLLM, Azure AI Foundry, GCP Vertex, and similar services.
    Sends an ``Authorization: Bearer`` header when *api_key_env* is set.
    """

    def __init__(self, endpoint: str, api_key_env: str | None = None) -> None:
        self.endpoint = endpoint
        self.api_key_env = api_key_env

    def call_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
        tools: list[dict] | None = None,
    ) -> dict:
        """Single API call. Returns the full response message dict.

        May contain ``tool_calls`` when *tools* are provided and the model
        chooses to invoke one.
        """
        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.endpoint.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str:
        return self.call_raw(model, messages, max_tokens, timeout).get("content") or ""
