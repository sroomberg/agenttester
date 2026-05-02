"""Shared fixtures for agenttester tests."""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from agenttester.config import AgentConfig


@pytest.fixture()
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo with an initial commit."""
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as cfg:
        cfg.set_value("user", "email", "test@test.com")
        cfg.set_value("user", "name", "Test")
    (tmp_path / "README.md").write_text("# test repo\n")
    repo.index.add(["README.md"])
    repo.index.commit("initial")
    return tmp_path


@pytest.fixture()
def echo_agent() -> AgentConfig:
    """An agent that echoes the prompt to a file (for testing)."""
    return AgentConfig(
        name="echo-agent",
        command="bash -c 'echo {prompt} > output.txt'",
        commit_style="manual",
        timeout=10,
    )


@pytest.fixture()
def stdin_agent() -> AgentConfig:
    """An agent that reads stdin and writes it to a file."""
    return AgentConfig(
        name="stdin-agent",
        command="bash -c 'cat > output.txt'",
        commit_style="manual",
        timeout=10,
    )
