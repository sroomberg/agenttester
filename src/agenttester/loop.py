"""Agent loop for tool-use task execution."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from .questions import QuestionRegistry
from .tools import ToolExecutor

_DEFAULT_MAX_TURNS = 20

# Patterns for XML-style tool calls models emit in text
# Format 1: <function=name>\n<parameter=key>value</parameter>\n</function>
_FUNC_EQ_RE = re.compile(
    r"<function=(\w+)>\s*"
    r"((?:<parameter=\w+>[\s\S]*?</parameter>\s*)*)"
    r"</function>",
    re.DOTALL,
)
_PARAM_EQ_RE = re.compile(r"<parameter=(\w+)>([\s\S]*?)</parameter>")

# Format 2: <function_calls><invoke name="x">...</invoke></function_calls>
_INVOKE_RE = re.compile(
    r'<invoke\s+name="(\w+)">\s*'
    r"((?:<parameter[^>]*>[^<]*</parameter>\s*)*)"
    r"</invoke>",
    re.DOTALL,
)
_PARAM_NAME_RE = re.compile(r'<parameter\s+name="(\w+)">([^<]*)</parameter>')

# Format 3: ```bash\n...\n``` (markdown code blocks used as tool calls)
_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)


def _parse_text_tool_calls(text: str) -> list[dict] | None:
    """Extract tool calls from XML/markdown patterns in model text output.

    Returns a list of OpenAI-format tool call dicts, or None if none found.
    """
    calls: list[dict] = []

    # Format 1: <function=bash><parameter=command>...</parameter></function>
    for match in _FUNC_EQ_RE.finditer(text):
        name = match.group(1)
        params_raw = match.group(2)
        params = {k: v.strip() for k, v in _PARAM_EQ_RE.findall(params_raw)}
        calls.append(
            {
                "id": f"text_{uuid.uuid4().hex[:12]}",
                "function": {"name": name, "arguments": json.dumps(params)},
            }
        )

    if calls:
        return calls

    # Format 2: <invoke name="bash"><parameter name="command">...</parameter></invoke>
    for match in _INVOKE_RE.finditer(text):
        name = match.group(1)
        params_raw = match.group(2)
        params = {k: v.strip() for k, v in _PARAM_NAME_RE.findall(params_raw)}
        calls.append(
            {
                "id": f"text_{uuid.uuid4().hex[:12]}",
                "function": {"name": name, "arguments": json.dumps(params)},
            }
        )

    if calls:
        return calls

    # Format 3: ```bash\n...\n``` markdown code blocks
    for match in _BASH_BLOCK_RE.finditer(text):
        command = match.group(1).strip()
        if command:
            calls.append(
                {
                    "id": f"text_{uuid.uuid4().hex[:12]}",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }
            )

    return calls if calls else None


async def run_agent_loop(
    provider: Any,
    model_id: str,
    messages: list[dict],
    prompt: str,
    executor: ToolExecutor,
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_tokens: int = 4096,
    on_event: Callable[[str, str], None] | None = None,
    question_registry: QuestionRegistry | None = None,
    model_name: str | None = None,
) -> str:
    """Async tool-use agent loop, mutating *messages* in place.

    Appends the user message, all assistant/tool turns, and the final
    assistant response to *messages*.  Returns the final text response.

    When max_turns is exhausted and a question_registry is provided, the
    loop pauses and asks the user for a continuation prompt rather than
    giving up.

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

    turns_used = 0
    while True:
        for _ in range(max_turns - turns_used):
            turns_used += 1
            msg = await provider.async_stream_raw(
                model_id,
                messages,
                max_tokens,
                tools=executor.tool_definitions,
                on_chunk=_on_chunk,
            )
            tool_calls = msg.get("tool_calls")
            text_based = False

            # Fallback: parse XML tool calls from text for models that
            # don't use structured tool_use (e.g. qwen, some open models)
            if not tool_calls:
                text = msg.get("content") or ""
                tool_calls = _parse_text_tool_calls(text)
                if not tool_calls:
                    messages.append({"role": "assistant", "content": text})
                    if on_event:
                        on_event("text", text)
                    return text
                text_based = True

            # Record assistant message
            if text_based:
                messages.append(
                    {"role": "assistant", "content": msg.get("content") or ""}
                )
            else:
                assistant_msg: dict = {"role": "assistant", "tool_calls": tool_calls}
                if msg.get("content"):
                    assistant_msg["content"] = msg["content"]
                messages.append(assistant_msg)

            # Execute each tool call
            results: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                tool_id = tc.get("id", "")
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                if on_event:
                    on_event("tool_call", f"{tool_name}: {fn.get('arguments', '')}")

                try:
                    result = await asyncio.to_thread(
                        executor.execute, tool_name, arguments
                    )
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"

                if on_event:
                    on_event("tool_result", result)

                if not tool_id:
                    tool_id = f"call_{tool_name}_{id(tc)}"

                results.append(f"[{tool_name}] {result or '(no output)'}")

                if not text_based:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result or "(no output)",
                        }
                    )

            # For text-based tool calls, feed results back as a user message
            if text_based:
                results_text = "\n\n".join(results)
                messages.append(
                    {"role": "user", "content": f"Tool results:\n{results_text}"}
                )

        # Max turns exhausted — ask user whether to continue
        if question_registry and model_name:
            if on_event:
                on_event(
                    "status",
                    f"reached {turns_used} turns — waiting for instructions",
                )
            continuation = await asyncio.to_thread(
                question_registry.ask,
                model_name,
                f"Reached {turns_used} tool turns without a final response. "
                "Send a message to continue, or reply 'stop' to end.",
            )
            if continuation is None:
                # Timed out or session exiting — stop cleanly without
                # polluting message history. The conversation can be
                # resumed later with a new prompt.
                return ""
            if continuation.lower().strip() in ("stop", "quit", "exit"):
                return ""
            messages.append({"role": "user", "content": continuation})
            turns_used = 0
        else:
            return ""
