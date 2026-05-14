"""Tests for agenttester.tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agenttester.tools import ToolExecutor, _truncate

# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_output_unchanged(self) -> None:
        assert _truncate("hello", max_bytes=100) == "hello"

    def test_long_output_truncated(self) -> None:
        big = "x" * 10000
        result = _truncate(big, max_bytes=100)
        assert "[output truncated]" in result
        assert len(result.encode()) < len(big.encode())

    def test_truncated_keeps_start_and_end(self) -> None:
        big = "START" + "m" * 10000 + "END"
        result = _truncate(big, max_bytes=100)
        assert "START" in result
        assert "END" in result


# ---------------------------------------------------------------------------
# ToolExecutor.execute dispatch
# ---------------------------------------------------------------------------


class TestExecuteDispatch:
    def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("nonexistent", {})
        assert "Unknown tool" in result

    def test_bad_arguments_returns_error(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("bash", {"wrong_key": "value"})
        assert "Error" in result


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class TestBash:
    def test_captures_stdout(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("bash", {"command": "echo hello"})
        assert "hello" in result

    def test_captures_stderr(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("bash", {"command": "echo err >&2"})
        assert "err" in result

    def test_nonzero_exit_prefixes_error(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("bash", {"command": "exit 1"})
        assert "Error (exit 1)" in result

    def test_empty_output_returns_placeholder(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("bash", {"command": "true"})
        assert result == "(no output)"

    def test_runs_in_workdir(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("bash", {"command": "pwd"})
        assert str(tmp_path) in result


# ---------------------------------------------------------------------------
# read_file / write_file
# ---------------------------------------------------------------------------


class TestReadWriteFile:
    def test_write_then_read(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        ex.execute("write_file", {"path": "hello.txt", "content": "world"})
        result = ex.execute("read_file", {"path": "hello.txt"})
        assert result == "world"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        ex.execute("write_file", {"path": "a/b/c.txt", "content": "hi"})
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "hi"

    def test_write_returns_char_count(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("write_file", {"path": "f.txt", "content": "abc"})
        assert "3" in result

    def test_read_missing_file_returns_error(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        result = ex.execute("read_file", {"path": "missing.txt"})
        assert "Error" in result

    def test_absolute_path_is_respected(self, tmp_path: Path) -> None:
        target = tmp_path / "abs.txt"
        target.write_text("absolute")
        ex = ToolExecutor(workdir=str(tmp_path / "other"))
        result = ex.execute("read_file", {"path": str(target)})
        assert result == "absolute"


# ---------------------------------------------------------------------------
# git_clone
# ---------------------------------------------------------------------------


class TestGitClone:
    def test_calls_git_clone(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Cloning…", stderr=""
            )
            ex.execute("git_clone", {"url": "https://github.com/org/repo.git"})
        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["git", "clone"]
        assert "https://github.com/org/repo.git" in cmd

    def test_branch_flag_included(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            ex.execute(
                "git_clone",
                {"url": "https://example.com/r.git", "branch": "main"},
            )
        cmd = mock_run.call_args[0][0]
        assert "--branch" in cmd
        assert "main" in cmd

    def test_pem_sets_git_ssh_command(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path), pem_path="/key.pem")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            ex.execute("git_clone", {"url": "git@github.com:org/repo.git"})
        env = mock_run.call_args[1]["env"]
        assert "GIT_SSH_COMMAND" in env
        assert "/key.pem" in env["GIT_SSH_COMMAND"]


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------


class TestGitCommit:
    def test_runs_add_then_commit(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            ex.execute("git_commit", {"message": "my commit"})

        assert calls[0] == ["git", "add", "-A"]
        assert calls[1][:3] == ["git", "commit", "-m"]
        assert "my commit" in calls[1]

    def test_stops_if_add_fails(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=1, stdout="", stderr="fatal error")

        with patch("subprocess.run", side_effect=fake_run):
            result = ex.execute("git_commit", {"message": "msg"})

        assert len(calls) == 1
        assert "Error" in result


# ---------------------------------------------------------------------------
# git_push
# ---------------------------------------------------------------------------


class TestGitPush:
    def test_calls_git_push(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="pushed", stderr="")
            ex.execute("git_push", {"branch": "my-branch"})
        cmd = mock_run.call_args[0][0]
        assert cmd == ["git", "push", "origin", "my-branch"]

    def test_custom_remote(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            ex.execute("git_push", {"branch": "feat", "remote": "upstream"})
        cmd = mock_run.call_args[0][0]
        assert "upstream" in cmd

    def test_pem_sets_git_ssh_command(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path), pem_path="/my.pem")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            ex.execute("git_push", {"branch": "main"})
        env = mock_run.call_args[1]["env"]
        assert "/my.pem" in env["GIT_SSH_COMMAND"]

    def test_no_pem_no_ssh_override(self, tmp_path: Path) -> None:
        ex = ToolExecutor(workdir=str(tmp_path))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            ex.execute("git_push", {"branch": "main"})
        env = mock_run.call_args[1]["env"]
        assert "GIT_SSH_COMMAND" not in env
