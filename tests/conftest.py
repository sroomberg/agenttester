"""Shared fixtures for agenttester tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agenttester.config import AgentConfig


@pytest.fixture()
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo with an initial commit."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    # Create an initial file and commit
    (tmp_path / "README.md").write_text("# test repo\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
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
