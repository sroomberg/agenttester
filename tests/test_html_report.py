"""Tests for HTML report generation."""

from __future__ import annotations

from unittest.mock import MagicMock

from agenttester.agent_runner import AgentResult
from agenttester.git_manager import DiffStats
from agenttester.html_report import generate_html_report


def test_html_contains_summary_table() -> None:
    git = MagicMock()
    git.get_diff_stats.return_value = DiffStats(
        files_changed=2, insertions=10, deletions=1, changed_files=["x.py"]
    )
    results = [
        AgentResult("claude", 0, 3.5, "", "", None),
        AgentResult("cursor", 1, 2.0, "", "", "timeout"),
    ]
    html = generate_html_report(
        "demo-run",
        "deadbeef" * 5,
        "fix bug",
        results,
        git,
    )
    assert "<!DOCTYPE html>" in html
    assert "claude" in html
    assert "cursor" in html
    assert "fix bug" in html
    assert "<table>" in html
