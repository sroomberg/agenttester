"""Tests for agenttester.loop."""

from __future__ import annotations

from unittest.mock import MagicMock

from agenttester.loop import run_agent_loop
from agenttester.tools import ToolExecutor


def _make_provider(responses: list[dict]) -> MagicMock:
    provider = MagicMock()
    # stream_raw returns the same normalized dict as call_raw; on_chunk is ignored
    provider.stream_raw.side_effect = [
        # wrap so on_chunk kwarg is accepted and ignored
        r
        for r in responses
    ]
    return provider


def _make_executor() -> ToolExecutor:
    ex = MagicMock(spec=ToolExecutor)
    ex.execute.return_value = "tool result"
    return ex


# ---------------------------------------------------------------------------
# Final text response (no tool calls)
# ---------------------------------------------------------------------------


class TestFinalText:
    def test_returns_text_when_no_tool_calls(self) -> None:
        provider = _make_provider([{"content": "hello", "tool_calls": None}])
        messages: list[dict] = []
        result = run_agent_loop(provider, "m", messages, "hi", _make_executor())
        assert result == "hello"

    def test_appends_user_and_assistant_to_messages(self) -> None:
        provider = _make_provider([{"content": "reply", "tool_calls": None}])
        messages: list[dict] = []
        run_agent_loop(provider, "m", messages, "question", _make_executor())
        assert messages[0] == {"role": "user", "content": "question"}
        assert messages[1] == {"role": "assistant", "content": "reply"}

    def test_preserves_existing_messages(self) -> None:
        provider = _make_provider([{"content": "ok", "tool_calls": None}])
        messages: list[dict] = [{"role": "system", "content": "you are helpful"}]
        run_agent_loop(provider, "m", messages, "q", _make_executor())
        assert messages[0]["role"] == "system"

    def test_on_event_text_called(self) -> None:
        provider = _make_provider([{"content": "final", "tool_calls": None}])
        events: list[tuple] = []
        run_agent_loop(
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

    def test_executes_tool_and_continues(self) -> None:
        provider = _make_provider(
            [
                self._tool_call(),
                {"content": "done", "tool_calls": None},
            ]
        )
        executor = _make_executor()
        messages: list[dict] = []
        result = run_agent_loop(provider, "m", messages, "do it", executor)
        assert result == "done"
        executor.execute.assert_called_once_with("bash", {"command": "ls"})

    def test_tool_result_appended_to_messages(self) -> None:
        provider = _make_provider(
            [
                self._tool_call(),
                {"content": "done", "tool_calls": None},
            ]
        )
        executor = _make_executor()
        executor.execute.return_value = "ls output"
        messages: list[dict] = []
        run_agent_loop(provider, "m", messages, "q", executor)
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert tool_msg["content"] == "ls output"
        assert tool_msg["tool_call_id"] == "call-1"

    def test_on_event_tool_call_and_result_called(self) -> None:
        provider = _make_provider(
            [
                self._tool_call("bash", '{"command":"ls"}'),
                {"content": "ok", "tool_calls": None},
            ]
        )
        events: list[tuple] = []
        run_agent_loop(
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

    def test_invalid_json_arguments_dont_crash(self) -> None:
        bad_args = {
            "content": None,
            "tool_calls": [
                {"id": "x", "function": {"name": "bash", "arguments": "not-json"}}
            ],
        }
        provider = _make_provider([bad_args, {"content": "ok", "tool_calls": None}])
        executor = _make_executor()
        result = run_agent_loop(provider, "m", [], "q", executor)
        assert result == "ok"
        executor.execute.assert_called_once_with("bash", {})


# ---------------------------------------------------------------------------
# Max turns
# ---------------------------------------------------------------------------


class TestMaxTurns:
    def test_returns_sentinel_when_max_turns_reached(self) -> None:
        always_tool = {
            "content": None,
            "tool_calls": [
                {"id": "c", "function": {"name": "bash", "arguments": "{}"}}
            ],
        }
        provider = _make_provider([always_tool] * 25)
        result = run_agent_loop(provider, "m", [], "q", _make_executor(), max_turns=3)
        assert "max turns" in result

    def test_sentinel_appended_to_messages(self) -> None:
        always_tool = {
            "content": None,
            "tool_calls": [
                {"id": "c", "function": {"name": "bash", "arguments": "{}"}}
            ],
        }
        provider = _make_provider([always_tool] * 25)
        messages: list[dict] = []
        run_agent_loop(provider, "m", messages, "q", _make_executor(), max_turns=2)
        last = messages[-1]
        assert last["role"] == "assistant"
        assert "max turns" in last["content"]
