"""Tests for agenttester.agent_runner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from agenttester.agent_runner import _prepare_command, run_agent
from agenttester.config import AgentConfig


@pytest.fixture()
def console() -> Console:
    return Console(quiet=True)


@pytest.fixture()
def lock() -> asyncio.Lock:
    return asyncio.Lock()


def _make_readline(lines: list[bytes]) -> AsyncMock:
    """Create an async readline that yields lines then b'' forever."""
    data = [*lines, b""]
    idx = 0

    async def _readline() -> bytes:
        nonlocal idx
        if idx < len(data):
            val = data[idx]
            idx += 1
            return val
        return b""

    return AsyncMock(side_effect=_readline)


def _make_mock_proc(
    returncode: int = 0,
    stdout_lines: list[bytes] | None = None,
    stderr_lines: list[bytes] | None = None,
) -> AsyncMock:
    """Build a mock asyncio.subprocess.Process."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.pid = 12345

    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    proc.stdout.readline = _make_readline(stdout_lines or [])
    proc.stderr.readline = _make_readline(stderr_lines or [])

    proc.stdin = None
    proc.wait = AsyncMock(return_value=returncode)
    proc.kill = MagicMock()
    return proc


# ── _prepare_command (pure function) ──────────────────────────────────


class TestPrepareCommand:
    def test_substitutes_inline_prompt(self) -> None:
        agent = AgentConfig(name="t", command="echo {prompt}")
        cmd, prompt_file, pipe_stdin = _prepare_command(agent, "hello world")
        assert cmd == "echo 'hello world'"
        assert prompt_file is None
        assert not pipe_stdin

    def test_escapes_special_characters(self) -> None:
        agent = AgentConfig(name="t", command="echo {prompt}")
        cmd, _, _ = _prepare_command(agent, "it's a \"test\"")
        assert "it" in cmd
        assert "test" in cmd

    def test_substitutes_prompt_file(self) -> None:
        agent = AgentConfig(name="t", command="cat {prompt_file}")
        cmd, prompt_file, pipe_stdin = _prepare_command(agent, "content")
        assert prompt_file is not None
        assert prompt_file.exists()
        assert prompt_file.read_text() == "content"
        assert str(prompt_file) in cmd
        assert not pipe_stdin
        prompt_file.unlink()

    def test_stdin_mode_when_no_placeholder(self) -> None:
        agent = AgentConfig(name="t", command="my-agent --run")
        cmd, prompt_file, pipe_stdin = _prepare_command(agent, "go")
        assert cmd == "my-agent --run"
        assert prompt_file is None
        assert pipe_stdin

    def test_both_placeholders(self) -> None:
        agent = AgentConfig(
            name="t", command="agent {prompt} --file {prompt_file}"
        )
        cmd, prompt_file, pipe_stdin = _prepare_command(agent, "hello")
        assert "hello" in cmd
        assert prompt_file is not None
        assert not pipe_stdin
        prompt_file.unlink()


# ── run_agent with mocked subprocess ──────────────────────────────────


class TestRunAgentLocal:
    @pytest.mark.asyncio
    async def test_success(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(name="test", command="echo {prompt}", timeout=10)
        proc = _make_mock_proc(returncode=0, stdout_lines=[b"hello\n"])

        with patch(
            "agenttester.agent_runner.asyncio.create_subprocess_shell",
            return_value=proc,
        ):
            result = await run_agent(
                agent, tmp_path, "hello", console, "cyan", lock
            )

        assert result.exit_code == 0
        assert result.error is None
        assert result.agent_name == "test"
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_captures_stdout(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(name="test", command="echo {prompt}", timeout=10)
        proc = _make_mock_proc(
            returncode=0, stdout_lines=[b"line1\n", b"line2\n"]
        )

        with patch(
            "agenttester.agent_runner.asyncio.create_subprocess_shell",
            return_value=proc,
        ):
            result = await run_agent(
                agent, tmp_path, "go", console, "cyan", lock
            )

        assert "line1" in result.stdout
        assert "line2" in result.stdout

    @pytest.mark.asyncio
    async def test_captures_stderr(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(name="test", command="echo {prompt}", timeout=10)
        proc = _make_mock_proc(returncode=0, stderr_lines=[b"warning\n"])

        with patch(
            "agenttester.agent_runner.asyncio.create_subprocess_shell",
            return_value=proc,
        ):
            result = await run_agent(
                agent, tmp_path, "go", console, "cyan", lock
            )

        assert "warning" in result.stderr

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(name="test", command="failing-cmd", timeout=10)
        proc = _make_mock_proc(returncode=42)

        with patch(
            "agenttester.agent_runner.asyncio.create_subprocess_shell",
            return_value=proc,
        ):
            result = await run_agent(
                agent, tmp_path, "", console, "cyan", lock
            )

        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_stdin_sends_prompt(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(name="test", command="my-agent", timeout=10)
        proc = _make_mock_proc(returncode=0)
        stdin = AsyncMock()
        stdin.write = MagicMock()
        stdin.drain = AsyncMock()
        stdin.close = MagicMock()
        proc.stdin = stdin

        with patch(
            "agenttester.agent_runner.asyncio.create_subprocess_shell",
            return_value=proc,
        ):
            result = await run_agent(
                agent, tmp_path, "prompt data", console, "cyan", lock
            )

        assert result.exit_code == 0
        stdin.write.assert_called_once_with(b"prompt data")

    @pytest.mark.asyncio
    async def test_called_with_correct_cwd(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(name="test", command="echo {prompt}", timeout=10)
        proc = _make_mock_proc(returncode=0)

        with patch(
            "agenttester.agent_runner.asyncio.create_subprocess_shell",
            return_value=proc,
        ) as mock_create:
            await run_agent(agent, tmp_path, "go", console, "cyan", lock)

        _, kwargs = mock_create.call_args
        assert kwargs["cwd"] == tmp_path

    @pytest.mark.asyncio
    async def test_env_merged(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(
            name="test",
            command="echo {prompt}",
            env={"MY_KEY": "val"},
            timeout=10,
        )
        proc = _make_mock_proc(returncode=0)

        with patch(
            "agenttester.agent_runner.asyncio.create_subprocess_shell",
            return_value=proc,
        ) as mock_create:
            await run_agent(agent, tmp_path, "go", console, "cyan", lock)

        _, kwargs = mock_create.call_args
        assert kwargs["env"]["MY_KEY"] == "val"


class TestRunAgentTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_error(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(name="slow", command="sleep 999", timeout=1)
        proc = _make_mock_proc(returncode=None)

        with (
            patch(
                "agenttester.agent_runner.asyncio.create_subprocess_shell",
                return_value=proc,
            ),
            patch(
                "agenttester.agent_runner._stream_and_wait",
                side_effect=TimeoutError,
            ),
            patch("agenttester.agent_runner._kill_proc_tree") as mock_kill,
        ):
            result = await run_agent(
                agent, tmp_path, "go", console, "cyan", lock
            )

        assert result.exit_code == -1
        assert "Timed out" in result.error
        mock_kill.assert_called_once()


class TestRunAgentRemote:
    @pytest.mark.asyncio
    async def test_dispatches_to_ssh(
        self, tmp_path: Path, console: Console, lock: asyncio.Lock
    ) -> None:
        agent = AgentConfig(
            name="remote-test",
            command="echo {prompt}",
            host="user@server",
            timeout=10,
        )
        proc = _make_mock_proc(returncode=0)

        with (
            patch("agenttester.agent_runner._rsync_to_remote"),
            patch("agenttester.agent_runner._rsync_from_remote"),
            patch(
                "agenttester.agent_runner.asyncio.create_subprocess_shell",
                return_value=proc,
            ) as mock_create,
        ):
            result = await run_agent(
                agent, tmp_path, "go", console, "cyan", lock
            )

        assert result.exit_code == 0
        cmd_str = mock_create.call_args[0][0]
        assert "ssh user@server" in cmd_str

    def test_is_remote_property(self) -> None:
        local = AgentConfig(name="l", command="x", host="localhost")
        remote = AgentConfig(name="r", command="x", host="user@box")
        assert not local.is_remote
        assert remote.is_remote
