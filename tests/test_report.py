"""Tests for agenttester.report."""

from __future__ import annotations

from unittest.mock import MagicMock

from agenttester.agent_runner import AgentResult
from agenttester.evaluator import EvaluatorResult
from agenttester.git_manager import DiffStats
from agenttester.report import generate_report


def _mock_git(stats_by_agent: dict[str, DiffStats] | None = None) -> MagicMock:
    """Create a mock GitManager that returns given diff stats."""
    git = MagicMock()
    default_stats = DiffStats(
        files_changed=2, insertions=10, deletions=3, changed_files=["a.py", "b.py"]
    )
    if stats_by_agent:
        git.get_diff_stats.side_effect = lambda agent_name, run_name, base_ref: (
            stats_by_agent.get(agent_name, default_stats)
        )
    else:
        git.get_diff_stats.return_value = default_stats
    return git


class TestGenerateReport:
    def test_contains_header(self) -> None:
        result = AgentResult("agent1", 0, 5.0, "out", "", None)
        report = generate_report(
            "abc123",
            "deadbeef" * 5,
            "do stuff",
            [result],
            _mock_git(),
        )
        assert "# AgentTester Report: abc123" in report

    def test_contains_prompt(self) -> None:
        result = AgentResult("agent1", 0, 1.0, "", "", None)
        report = generate_report("r1", "a" * 40, "fix the bug", [result], _mock_git())
        assert "fix the bug" in report

    def test_contains_summary_table(self) -> None:
        results = [
            AgentResult("claude", 0, 10.2, "", "", None),
            AgentResult("aider", 1, 5.5, "", "", "crashed"),
        ]
        report = generate_report("r2", "b" * 40, "test", results, _mock_git())
        assert "| claude |" in report
        assert "| aider |" in report
        assert "✅" in report
        assert "❌" in report

    def test_per_agent_sections(self) -> None:
        results = [
            AgentResult("agent1", 0, 3.0, "", "", None),
            AgentResult("agent2", 0, 4.0, "", "", None),
        ]
        report = generate_report("r3", "c" * 40, "test", results, _mock_git())
        assert "## agent1" in report
        assert "## agent2" in report
        assert "`agenttester/agent1/r3`" in report

    def test_includes_changed_files(self) -> None:
        result = AgentResult("agent1", 0, 1.0, "", "", None)
        report = generate_report("r4", "d" * 40, "test", [result], _mock_git())
        assert "`a.py`" in report
        assert "`b.py`" in report

    def test_includes_diff_stats_in_table(self) -> None:
        result = AgentResult("agent1", 0, 1.0, "", "", None)
        stats = DiffStats(files_changed=5, insertions=20, deletions=8, changed_files=[])
        git = _mock_git({"agent1": stats})
        report = generate_report("r5", "e" * 40, "test", [result], git)
        assert "+20" in report
        assert "-8" in report

    def test_error_shown_in_report(self) -> None:
        result = AgentResult("agent1", -1, 1.0, "", "", "Timed out after 10s")
        report = generate_report("r6", "f" * 40, "test", [result], _mock_git())
        assert "Timed out after 10s" in report

    def test_iteration_shown_in_report(self) -> None:
        result = AgentResult("agent1", 0, 1.0, "", "", None)
        report = generate_report(
            "r7", "g" * 40, "test", [result], _mock_git(), iteration=3
        )
        assert "**Iteration**: 3" in report

    def test_eval_results_shown_per_agent(self) -> None:
        result = AgentResult("agent1", 0, 1.0, "", "", None)
        ev = EvaluatorResult("llama3", "agent1", "Looks good overall.", 2.5)
        report = generate_report(
            "r8",
            "h" * 40,
            "test",
            [result],
            _mock_git(),
            eval_results={"agent1": [ev]},
        )
        assert "### Evaluations" in report
        assert "llama3" in report
        assert "Looks good overall." in report

    def test_aggregate_shown_per_agent(self) -> None:
        result = AgentResult("agent1", 0, 1.0, "", "", None)
        report = generate_report(
            "r9",
            "i" * 40,
            "test",
            [result],
            _mock_git(),
            aggregates={"agent1": "Aggregate: solid work."},
        )
        assert "### Aggregate Assessment" in report
        assert "Aggregate: solid work." in report

    def test_no_eval_sections_when_not_provided(self) -> None:
        result = AgentResult("agent1", 0, 1.0, "", "", None)
        report = generate_report("r10", "j" * 40, "test", [result], _mock_git())
        assert "### Evaluations" not in report
        assert "### Aggregate Assessment" not in report
