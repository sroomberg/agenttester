"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable

from .base import Provider


class OpenAICompatProvider(Provider):
    """Calls any OpenAI-compatible chat completions endpoint.

    Covers vLLM, Azure AI Foundry, GCP Vertex, and similar services.
    Sends an ``Authorization: Bearer`` header when *api_key_env* is set.
    """

    def __init__(self, endpoint: str, api_key_env: str | None = None) -> None:
        self.endpoint = endpoint
        self.api_key_env = api_key_env

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def call_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
        tools: list[dict] | None = None,
    ) -> dict:
        """Single (non-streaming) API call. Returns the full response message dict.

        May contain ``tool_calls`` when *tools* are provided and the model
        chooses to invoke one.
        """
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.endpoint.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=self._headers(),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]

    def stream_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
        tools: list[dict] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        """Streaming API call via SSE. Calls *on_chunk* for each text chunk.

        Returns the same normalized dict as ``call_raw`` (with ``content`` and
        optional ``tool_calls``) once the stream is exhausted.
        """
        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.endpoint.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=self._headers(),
        )

        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                chunk_text = delta.get("content") or ""
                if chunk_text:
                    text_parts.append(chunk_text)
                    if on_chunk:
                        on_chunk(chunk_text)

                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.get("id"):
                        tool_calls_acc[idx]["id"] = tc_delta["id"]
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        tool_calls_acc[idx]["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

        text = "".join(text_parts)
        tool_calls = (
            [tool_calls_acc[k] for k in sorted(tool_calls_acc)]
            if tool_calls_acc
            else None
        )
        return {"content": text or None, "tool_calls": tool_calls}

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str:
        return self.call_raw(model, messages, max_tokens, timeout).get("content") or ""
