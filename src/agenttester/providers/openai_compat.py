"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import aiohttp

from .base import _API_CALL_TIMEOUT, _API_STREAM_TIMEOUT, Provider


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

    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str:
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        async with (
            aiohttp.ClientSession(timeout=_API_CALL_TIMEOUT) as session,
            session.post(
                f"{self.endpoint.rstrip('/')}/v1/chat/completions",
                json=body,
                headers=self._headers(),
            ) as resp,
        ):
            data = await resp.json()
        return data["choices"][0]["message"].get("content") or ""

    async def async_stream_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        """Async SSE streaming call via aiohttp. No read timeout — streams until done.

        Calls *on_chunk* for each text chunk as it arrives. Returns the same
        normalized dict as ``call_raw`` once the stream is exhausted.
        """
        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}

        async with (
            aiohttp.ClientSession(timeout=_API_STREAM_TIMEOUT) as session,
            session.post(
                f"{self.endpoint.rstrip('/')}/v1/chat/completions",
                json=body,
                headers=self._headers(),
            ) as resp,
        ):
            buffer = ""
            while True:
                raw_line = await resp.content.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    buffer += payload
                    try:
                        data = json.loads(buffer)
                        buffer = ""
                    except json.JSONDecodeError:
                        continue
                else:
                    buffer = ""
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
