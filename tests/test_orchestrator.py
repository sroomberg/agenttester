"""Tests for agenttester.orchestrator."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from agenttester.orchestrator import _build_prompt, _user_input_router


class TestBuildPrompt:
    def test_includes_branch(self) -> None:
        result = _build_prompt("Fix the bug", "abc123", "claude", "")
        assert "`agenttester/abc123/claude`" in result

    def test_includes_run_id_and_agent_name(self) -> None:
        result = _build_prompt("test", "run123", "myagent", "")
        assert "agenttester/run123/myagent" in result

    def test_preserves_original_prompt(self) -> None:
        original = "This is a multi-line\nprompt with special chars: !@#$%"
        result = _build_prompt(original, "x", "y", "")
        assert original in result

    def test_skills_prepended_before_branch(self) -> None:
        result = _build_prompt("do the thing", "r1", "claude", "Always edit freely.")
        skills_pos = result.index("Always edit freely.")
        branch_pos = result.index("agenttester/r1/claude")
        prompt_pos = result.index("do the thing")
        assert skills_pos < branch_pos < prompt_pos

    def test_no_skills_still_includes_branch_and_prompt(self) -> None:
        result = _build_prompt("my task", "r1", "agent1", "")
        assert "agenttester/r1/agent1" in result
        assert "my task" in result

    def test_empty_skills_not_double_newlined(self) -> None:
        result = _build_prompt("task", "r1", "a1", "")
        assert not result.startswith("\n")


def _make_router(
    stdin_lines: list[str],
    queues: dict[str, asyncio.Queue] | None = None,
) -> tuple[dict[str, asyncio.Queue], Console, asyncio.Lock, asyncio.Event]:
    """Set up a router with mock stdin that yields *stdin_lines* then stops."""
    if queues is None:
        queues = {}
    console = Console(quiet=True)
    lock = asyncio.Lock()
    done = asyncio.Event()

    call_count = 0
    lines = list(stdin_lines)

    def fake_select(fds, _w, _x, timeout):
        nonlocal call_count
        call_count += 1
        if call_count <= len(lines):
            return [fds[0]], [], []
        done.set()
        return [], [], []

    readline_idx = 0

    def fake_readline():
        nonlocal readline_idx
        if readline_idx < len(lines):
            val = lines[readline_idx]
            readline_idx += 1
            return val
        return ""

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdin.readline.side_effect = fake_readline

    return queues, console, lock, done, mock_stdin, fake_select


class TestUserInputRouter:
    @pytest.mark.asyncio
    async def test_routes_message_to_correct_agent(self) -> None:
        queue_a: asyncio.Queue = asyncio.Queue()
        queue_b: asyncio.Queue = asyncio.Queue()
        queues = {"agent-a": queue_a, "agent-b": queue_b}
        queues, console, lock, done, mock_stdin, fake_select = _make_router(
            ["@agent-a: hello world\n"], queues
        )
        with (
            patch("agenttester.orchestrator.sys.stdin", mock_stdin),
            patch("agenttester.orchestrator.select.select", side_effect=fake_select),
        ):
            await _user_input_router(queues, console, lock, done)

        assert not queue_a.empty()
        assert queue_a.get_nowait() == "hello world"
        assert queue_b.empty()

    @pytest.mark.asyncio
    async def test_does_not_cross_contaminate_agents(self) -> None:
        queue_a: asyncio.Queue = asyncio.Queue()
        queue_b: asyncio.Queue = asyncio.Queue()
        queues = {"agent-a": queue_a, "agent-b": queue_b}
        queues, console, lock, done, mock_stdin, fake_select = _make_router(
            ["@agent-b: only for b\n"], queues
        )
        with (
            patch("agenttester.orchestrator.sys.stdin", mock_stdin),
            patch("agenttester.orchestrator.select.select", side_effect=fake_select),
        ):
            await _user_input_router(queues, console, lock, done)

        assert queue_a.empty()
        assert not queue_b.empty()
        assert queue_b.get_nowait() == "only for b"

    @pytest.mark.asyncio
    async def test_ignores_lines_without_at_prefix(self) -> None:
        queue_a: asyncio.Queue = asyncio.Queue()
        queues = {"agent-a": queue_a}
        queues, console, lock, done, mock_stdin, fake_select = _make_router(
            ["just a plain line\n"], queues
        )
        with (
            patch("agenttester.orchestrator.sys.stdin", mock_stdin),
            patch("agenttester.orchestrator.select.select", side_effect=fake_select),
        ):
            await _user_input_router(queues, console, lock, done)

        assert queue_a.empty()

    @pytest.mark.asyncio
    async def test_warns_on_unknown_agent(self) -> None:
        queue_a: asyncio.Queue = asyncio.Queue()
        queues = {"agent-a": queue_a}
        queues, console, lock, done, mock_stdin, fake_select = _make_router(
            ["@ghost: hello\n"], queues
        )
        output: list[str] = []
        console.print = lambda *a, **kw: output.append(str(a[0]))  # type: ignore[method-assign]
        with (
            patch("agenttester.orchestrator.sys.stdin", mock_stdin),
            patch("agenttester.orchestrator.select.select", side_effect=fake_select),
        ):
            await _user_input_router(queues, console, lock, done)

        assert queue_a.empty()
        assert any("ghost" in line for line in output)

    @pytest.mark.asyncio
    async def test_exits_immediately_when_not_a_tty(self) -> None:
        queues: dict[str, asyncio.Queue] = {"agent-a": asyncio.Queue()}
        console = Console(quiet=True)
        lock = asyncio.Lock()
        done = asyncio.Event()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        with patch("agenttester.orchestrator.sys.stdin", mock_stdin):
            await _user_input_router(queues, console, lock, done)
        # Should return without reading anything
        mock_stdin.readline.assert_not_called()
