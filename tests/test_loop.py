"""Tests for agenttester.loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from agenttester.loop import run_agent_loop
from agenttester.tools import ToolExecutor


def _make_provider(responses: list[dict]) -> MagicMock:
    provider = MagicMock()
    provider.async_stream_raw = AsyncMock(side_effect=responses)
    return provider


def _make_executor() -> ToolExecutor:
    ex = MagicMock(spec=ToolExecutor)
    ex.execute.return_value = "tool result"
    return ex


# ---------------------------------------------------------------------------
# Final text response (no tool calls)
# ---------------------------------------------------------------------------


class TestFinalText:
    async def test_returns_text_when_no_tool_calls(self) -> None:
        provider = _make_provider([{"content": "hello", "tool_calls": None}])
        messages: list[dict] = []
        result = await run_agent_loop(provider, "m", messages, "hi", _make_executor())
        assert result == "hello"

    async def test_appends_user_and_assistant_to_messages(self) -> None:
        provider = _make_provider([{"content": "reply", "tool_calls": None}])
        messages: list[dict] = []
        await run_agent_loop(provider, "m", messages, "question", _make_executor())
        assert messages[0] == {"role": "user", "content": "question"}
        assert messages[1] == {"role": "assistant", "content": "reply"}

    async def test_preserves_existing_messages(self) -> None:
        provider = _make_provider([{"content": "ok", "tool_calls": None}])
        messages: list[dict] = [{"role": "system", "content": "you are helpful"}]
        await run_agent_loop(provider, "m", messages, "q", _make_executor())
        assert messages[0]["role"] == "system"

    async def test_on_event_text_called(self) -> None:
        provider = _make_provider([{"content": "final", "tool_calls": None}])
        events: list[tuple] = []
        await run_agent_loop(
            provider,
            "m",
            [],
            "q",
            _make_executor(),
            on_event=lambda t, c: events.append((t, c)),
        )
        assert ("text", "final") in events


# ---------------------------------------------------------------------------
# Tool call → final text
# ---------------------------------------------------------------------------


class TestToolCallThenText:
    def _tool_call(
        self, tool_name: str = "bash", args: str = '{"command":"ls"}'
    ) -> dict:
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": tool_name, "arguments": args},
                }
            ],
        }

    async def test_executes_tool_and_continues(self) -> None:
        provider = _make_provider(
            [
                self._tool_call(),
                {"content": "done", "tool_calls": None},
            ]
        )
        executor = _make_executor()
        messages: list[dict] = []
        result = await run_agent_loop(provider, "m", messages, "do it", executor)
        assert result == "done"
        executor.execute.assert_called_once_with("bash", {"command": "ls"})

    async def test_tool_result_appended_to_messages(self) -> None:
        provider = _make_provider(
            [
                self._tool_call(),
                {"content": "done", "tool_calls": None},
            ]
        )
        executor = _make_executor()
        executor.execute.return_value = "ls output"
        messages: list[dict] = []
        await run_agent_loop(provider, "m", messages, "q", executor)
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert tool_msg["content"] == "ls output"
        assert tool_msg["tool_call_id"] == "call-1"

    async def test_on_event_tool_call_and_result_called(self) -> None:
        provider = _make_provider(
            [
                self._tool_call("bash", '{"command":"ls"}'),
                {"content": "ok", "tool_calls": None},
            ]
        )
        events: list[tuple] = []
        await run_agent_loop(
            provider,
            "m",
            [],
            "q",
            _make_executor(),
            on_event=lambda t, c: events.append((t, c)),
        )
        types = [e[0] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "text" in types

    async def test_invalid_json_arguments_dont_crash(self) -> None:
        bad_args = {
            "content": None,
            "tool_calls": [
                {"id": "x", "function": {"name": "bash", "arguments": "not-json"}}
            ],
        }
        provider = _make_provider([bad_args, {"content": "ok", "tool_calls": None}])
        executor = _make_executor()
        result = await run_agent_loop(provider, "m", [], "q", executor)
        assert result == "ok"
        executor.execute.assert_called_once_with("bash", {})


# ---------------------------------------------------------------------------
# Max turns
# ---------------------------------------------------------------------------


class TestMaxTurns:
    async def test_returns_empty_when_max_turns_reached_no_registry(self) -> None:
        always_tool = {
            "content": None,
            "tool_calls": [
                {"id": "c", "function": {"name": "bash", "arguments": "{}"}}
            ],
        }
        provider = _make_provider([always_tool] * 25)
        result = await run_agent_loop(
            provider, "m", [], "q", _make_executor(), max_turns=3
        )
        assert result == ""

    async def test_last_message_is_tool_result_when_max_turns_reached(self) -> None:
        always_tool = {
            "content": None,
            "tool_calls": [
                {"id": "c", "function": {"name": "bash", "arguments": "{}"}}
            ],
        }
        provider = _make_provider([always_tool] * 25)
        messages: list[dict] = []
        await run_agent_loop(
            provider, "m", messages, "q", _make_executor(), max_turns=2
        )
        last = messages[-1]
        assert last["role"] == "tool"


# ---------------------------------------------------------------------------
# max_tokens auto-continue
# ---------------------------------------------------------------------------


class TestMaxTokensAutoContinue:
    async def test_continues_after_max_tokens(self) -> None:
        truncated = {
            "content": "partial answer",
            "tool_calls": None,
            "stop_reason": "max_tokens",
        }
        final = {"content": "and the rest", "tool_calls": None}
        provider = _make_provider([truncated, final])
        result = await run_agent_loop(provider, "m", [], "q", _make_executor())
        assert result == "and the rest"

    async def test_continue_appends_continue_message(self) -> None:
        truncated = {
            "content": "part1",
            "tool_calls": None,
            "stop_reason": "max_tokens",
        }
        final = {"content": "part2", "tool_calls": None}
        provider = _make_provider([truncated, final])
        messages: list[dict] = []
        await run_agent_loop(provider, "m", messages, "q", _make_executor())
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert messages[2]["content"] == "Continue from where you left off."

    async def test_on_event_status_emitted_on_continue(self) -> None:
        truncated = {
            "content": "part",
            "tool_calls": None,
            "stop_reason": "max_tokens",
        }
        final = {"content": "done", "tool_calls": None}
        provider = _make_provider([truncated, final])
        events: list[tuple] = []

        def _on(k: str, v: str) -> None:
            events.append((k, v))

        await run_agent_loop(provider, "m", [], "q", _make_executor(), on_event=_on)
        assert any(k == "status" and "truncated" in v for k, v in events)

    async def test_no_continue_when_empty_content(self) -> None:
        truncated = {"content": "", "tool_calls": None, "stop_reason": "max_tokens"}
        provider = _make_provider([truncated])
        result = await run_agent_loop(provider, "m", [], "q", _make_executor())
        assert result == ""
