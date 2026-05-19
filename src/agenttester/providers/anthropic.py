"""Anthropic Messages API provider."""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import aiohttp

from .base import _API_CALL_TIMEOUT, _API_STREAM_TIMEOUT, Provider


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Anthropic format."""
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return result


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Return (system_text, non_system_messages) for Anthropic's API format."""
    parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return ("\n\n".join(parts) if parts else None), rest


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Anthropic format.

    Groups consecutive ``role: tool`` messages into a single user message
    with ``tool_result`` content blocks, and converts assistant ``tool_calls``
    to ``tool_use`` content blocks.
    """
    result: list[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "tool":
            tool_results: list[dict] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": messages[i]["tool_call_id"],
                        "content": messages[i]["content"],
                    }
                )
                i += 1
            result.append({"role": "user", "content": tool_results})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": args,
                    }
                )
            result.append({"role": "assistant", "content": content})
            i += 1
        else:
            result.append({"role": m["role"], "content": m["content"]})
            i += 1
    return result


class AnthropicProvider(Provider):
    """Calls the Anthropic Messages API directly."""

    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self.api_key_env = api_key_env

    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str:
        api_key = os.environ.get(self.api_key_env, "")
        system, non_system = _split_system(messages)
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_anthropic_messages(non_system),
        }
        if system:
            body["system"] = system
        async with (
            aiohttp.ClientSession(timeout=_API_CALL_TIMEOUT) as session,
            session.post(
                "https://api.anthropic.com/v1/messages",
                json=body,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            ) as resp,
        ):
            data = await resp.json()
        return "\n".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )

    async def async_stream_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        """Async SSE streaming call via aiohttp. No read timeout — streams until done.

        Returns the same normalized dict as ``call_raw`` once the stream ends.
        """
        api_key = os.environ.get(self.api_key_env, "")
        system, non_system = _split_system(messages)
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_anthropic_messages(non_system),
            "stream": True,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = _to_anthropic_tools(tools)

        # blocks[index] = {"type": "text"|"tool_use", "id": "", "name": "", "content": ""}  # noqa: E501
        blocks: dict[int, dict] = {}
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        input_tokens = 0
        output_tokens = 0

        async with (
            aiohttp.ClientSession(timeout=_API_STREAM_TIMEOUT) as session,
            session.post(
                "https://api.anthropic.com/v1/messages",
                json=body,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
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

                event_type = data.get("type", "")

                if event_type == "message_start":
                    usage = data.get("message", {}).get("usage", {})
                    input_tokens += usage.get("input_tokens", 0)

                elif event_type == "message_delta":
                    usage = data.get("usage", {})
                    output_tokens += usage.get("output_tokens", 0)

                elif event_type == "content_block_start":
                    idx = data.get("index", 0)
                    block = data.get("content_block", {})
                    blocks[idx] = {
                        "type": block.get("type", "text"),
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "content": "",
                    }

                elif event_type == "content_block_delta":
                    idx = data.get("index", 0)
                    delta = data.get("delta", {})
                    if idx in blocks:
                        if delta.get("type") == "text_delta":
                            chunk = delta.get("text", "")
                            blocks[idx]["content"] += chunk
                            text_parts.append(chunk)
                            if on_chunk and chunk:
                                on_chunk(chunk)
                        elif delta.get("type") == "input_json_delta":
                            blocks[idx]["content"] += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    idx = data.get("index", 0)
                    if idx in blocks and blocks[idx]["type"] == "tool_use":
                        b = blocks[idx]
                        tool_calls.append(
                            {
                                "id": b["id"],
                                "function": {
                                    "name": b["name"],
                                    "arguments": b["content"],
                                },
                            }
                        )

        text = "".join(text_parts)
        return {
            "content": text if text else None,
            "tool_calls": tool_calls if tool_calls else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
