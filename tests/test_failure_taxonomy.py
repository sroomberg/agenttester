"""Tests for failure taxonomy."""

from __future__ import annotations

from agenttester.agent_runner import AgentResult
from agenttester.failure_taxonomy import classify_agent_result
from agenttester.git_manager import DiffStats


def test_success_with_changes() -> None:
    result = AgentResult("claude", 0, 1.0, "", "", None)
    stats = DiffStats(files_changed=1, insertions=1, deletions=0, changed_files=["a"])
    assert classify_agent_result(result, stats).category == "success"


def test_no_changes() -> None:
    result = AgentResult("claude", 0, 1.0, "", "", None)
    stats = DiffStats()
    assert classify_agent_result(result, stats).category == "no_changes"


def test_timeout() -> None:
    result = AgentResult("claude", -1, 600.0, "", "", "Timed out after 600s")
    stats = DiffStats()
    assert classify_agent_result(result, stats).category == "timeout"
