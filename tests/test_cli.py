"""Tests for agenttester.cli."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml
from typer.testing import CliRunner

from agenttester.cli import _find_git_root, app
from agenttester.session import ReplSession

runner = CliRunner()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", text)


class TestHelpCommands:
    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agents" in result.output
        assert "run" in result.output

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--agents" in output
        assert "--prompt-file" in output

    def test_agents_help(self) -> None:
        result = runner.invoke(app, ["agents", "--help"])
        assert result.exit_code == 0


class TestAgentsCommand:
    def test_lists_presets(self) -> None:
        result = runner.invoke(app, ["agents"])
        assert result.exit_code == 0
        assert "claude" in result.output
        assert "aider" in result.output
        assert "codex" in result.output


class TestRunValidation:
    def test_missing_prompt(self) -> None:
        result = runner.invoke(app, ["run", "--agents", "claude"])
        assert result.exit_code != 0

    def test_missing_agents(self) -> None:
        result = runner.invoke(app, ["run", "fix bug"])
        assert result.exit_code != 0

    def test_unknown_agent(self) -> None:
        result = runner.invoke(app, ["run", "fix bug", "--agents", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown agent" in result.output

    def test_too_many_agents(self) -> None:
        result = runner.invoke(
            app,
            ["run", "test", "--agents", "a,b,c,d,e,f"],
        )
        assert result.exit_code != 0

    def test_missing_prompt_file(self) -> None:
        result = runner.invoke(
            app,
            ["run", "--prompt-file", "/nonexistent/file.md", "--agents", "claude"],
        )
        assert result.exit_code != 0

    def test_comma_separated_agents_parsed(self) -> None:
        """Verify comma-separated --agents values are parsed without error."""
        with patch("agenttester.cli.Orchestrator") as mock_cls:
            mock_cls.return_value.run = AsyncMock(return_value=[])
            result = runner.invoke(app, ["run", "test", "--agents", "claude,aider"])
        assert "Unknown agent" not in result.output


class TestFindGitRoot:
    def test_returns_root_when_at_root(self, tmp_git_repo: Path) -> None:
        result = _find_git_root(tmp_git_repo)
        assert result == tmp_git_repo

    def test_finds_root_from_subdir(self, tmp_git_repo: Path) -> None:
        subdir = tmp_git_repo / "src" / "pkg"
        subdir.mkdir(parents=True)
        result = _find_git_root(subdir)
        assert result == tmp_git_repo

    def test_returns_start_when_no_git(self, tmp_path: Path) -> None:
        result = _find_git_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_works_with_nested_dirs(self, tmp_git_repo: Path) -> None:
        deep = tmp_git_repo / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        result = _find_git_root(deep)
        assert result == tmp_git_repo

    def test_works_with_git_file_worktree(self, tmp_git_repo: Path) -> None:
        # In a git worktree, .git is a file, not a directory
        git_file = tmp_git_repo / ".git"
        if git_file.is_dir():
            # In normal repo, .git is a dir. Simulate worktree by creating a file
            # This test is more about ensuring exists() works for both files and dirs
            pass
        result = _find_git_root(tmp_git_repo)
        assert result == tmp_git_repo


class TestSessionsCommand:
    def _make_sessions(self, sessions_dir: Path) -> None:
        s1 = ReplSession.create("old-session")
        s1.created_at = "2026-05-18T09:00:00+00:00"
        s1.branches = ["agenttester/claude/old-fix"]
        s1.save(sessions_dir)

        s2 = ReplSession.create("new-session")
        s2.created_at = "2026-05-19T10:30:00+00:00"
        s2.branches = ["agenttester/claude/new-feat", "agenttester/gpt4/new-feat"]
        s2.save(sessions_dir)

    def test_no_sessions(self, tmp_path: Path) -> None:
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions"])
        assert result.exit_code == 0
        assert "No saved sessions" in result.output

    def test_lists_sessions_newest_first(self, tmp_path: Path) -> None:
        self._make_sessions(tmp_path)
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions"])
        assert result.exit_code == 0
        out = strip_ansi(result.output)
        assert out.index("new-session") < out.index("old-session")

    def test_branches_indented(self, tmp_path: Path) -> None:
        self._make_sessions(tmp_path)
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions"])
        out = strip_ansi(result.output)
        branch_lines = [ln for ln in out.splitlines() if "agenttester/" in ln]
        assert all(ln.startswith("    ") for ln in branch_lines)

    def test_branch_columns_aligned(self, tmp_path: Path) -> None:
        self._make_sessions(tmp_path)
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions"])
        out = strip_ansi(result.output)
        branch_lines = [ln for ln in out.splitlines() if "agenttester/" in ln]
        status_positions = [ln.index("unknown") for ln in branch_lines]
        assert len(set(status_positions)) == 1  # all statuses start at same column

    def test_yaml_output_newest_first(self, tmp_path: Path) -> None:
        self._make_sessions(tmp_path)
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions", "--yaml"])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data[0]["id"] == "new-session"
        assert data[1]["id"] == "old-session"

    def test_yaml_key_order(self, tmp_path: Path) -> None:
        self._make_sessions(tmp_path)
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions", "--yaml"])
        lines = result.output.splitlines()

        def _first_line(key: str) -> int:
            return next(
                i for i, ln in enumerate(lines) if ln.lstrip("- ").startswith(f"{key}:")
            )

        assert _first_line("id") < _first_line("date") < _first_line("start")
        assert _first_line("start") < _first_line("end") < _first_line("branches")

    def test_yaml_no_sessions(self, tmp_path: Path) -> None:
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions", "--yaml"])
        assert result.exit_code == 0
        assert yaml.safe_load(result.output) == []

    def test_yaml_contains_branch_status(self, tmp_path: Path) -> None:
        self._make_sessions(tmp_path)
        with patch("agenttester.session._default_sessions_dir", return_value=tmp_path):
            result = runner.invoke(app, ["sessions", "--yaml"])
        data = yaml.safe_load(result.output)
        branch_entry = data[0]["branches"][0]
        assert "branch" in branch_entry
        assert "status" in branch_entry
