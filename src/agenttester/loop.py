"""Agent loop for tool-use task execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from .tools import ToolExecutor


async def run_agent_loop(
    provider: Any,
    model_id: str,
    messages: list[dict],
    prompt: str,
    executor: ToolExecutor,
    max_turns: int = 20,
    on_event: Callable[[str, str], None] | None = None,
) -> str:
    """Async tool-use agent loop, mutating *messages* in place.

    Appends the user message, all assistant/tool turns, and the final
    assistant response to *messages*.  Returns the final text response.

    on_event(type, content) is called for observability:
        "chunk"       → streaming text chunk (fires many times per turn)
        "tool_call"   → "{tool_name}: {args_json}"
        "tool_result" → tool output string
        "text"        → final assistant text (full accumulated response)
    """
    messages.append({"role": "user", "content": prompt})

    def _on_chunk(chunk: str) -> None:
        if on_event:
            on_event("chunk", chunk)

    for _ in range(max_turns):
        msg = await provider.async_stream_raw(
            model_id,
            messages,
            4096,
            tools=executor.tool_definitions,
            on_chunk=_on_chunk,
        )
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            text = msg.get("content") or ""
            messages.append({"role": "assistant", "content": text})
            if on_event:
                on_event("text", text)
            return text

        assistant_msg: dict = {"role": "assistant", "tool_calls": tool_calls}
        if msg.get("content"):
            assistant_msg["content"] = msg["content"]
        messages.append(assistant_msg)

        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if on_event:
                on_event("tool_call", f"{tool_name}: {fn.get('arguments', '')}")

            result = await asyncio.to_thread(executor.execute, tool_name, arguments)

            if on_event:
                on_event("tool_result", result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                }
            )

    text = "[max turns reached without final response]"
    messages.append({"role": "assistant", "content": text})
    return text
