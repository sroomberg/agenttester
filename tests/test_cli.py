"""Tests for agenttester.cli."""

from __future__ import annotations

from typer.testing import CliRunner

from agenttester.cli import app

runner = CliRunner(env={"NO_COLOR": "1"})


class TestHelpCommands:
    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agents" in result.output
        assert "run" in result.output

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--agents" in result.output
        assert "--prompt-file" in result.output

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
