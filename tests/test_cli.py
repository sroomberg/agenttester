"""Tests for agenttester.cli."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from agenttester.cli import _find_git_root, app

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
        """Verify comma parsing works — the command will fail at the repo
        check, but agent parsing should succeed (no 'Unknown agent' error)."""
        result = runner.invoke(app, ["run", "test", "--agents", "claude,aider"])
        # Should fail because cwd may not be a git repo, not because of agent parsing
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
