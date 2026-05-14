"""Anthropic Messages API provider."""

from __future__ import annotations

import json
import os
import urllib.request

from .base import Provider


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


def _from_anthropic_response(data: dict) -> dict:
    """Convert an Anthropic Messages API response to a normalized dict.

    Returns ``{"content": str | None, "tool_calls": list | None}`` matching
    the shape produced by ``OpenAICompatProvider.call_raw``.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    return {
        "content": "\n".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls if tool_calls else None,
    }


class AnthropicProvider(Provider):
    """Calls the Anthropic Messages API directly."""

    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self.api_key_env = api_key_env

    def call_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
        tools: list[dict] | None = None,
    ) -> dict:
        """Single API call with optional tool use.

        Accepts *messages* and *tools* in OpenAI format; converts to Anthropic
        format before sending.  Returns a normalized dict matching the shape of
        ``OpenAICompatProvider.call_raw``: ``{"content": …, "tool_calls": …}``.
        """
        api_key = os.environ.get(self.api_key_env, "")
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_anthropic_messages(
                [m for m in messages if m["role"] != "system"]
            ),
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if tools:
            body["tools"] = _to_anthropic_tools(tools)
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return _from_anthropic_response(data)

    def call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        timeout: int = 120,
    ) -> str:
        return self.call_raw(model, messages, max_tokens, timeout).get("content") or ""
