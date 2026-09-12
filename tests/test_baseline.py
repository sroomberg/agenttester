"""Tests for baseline comparison."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agenttester.agent_runner import AgentResult
from agenttester.baseline import (
    BaselineFile,
    compare_run_to_baseline,
    format_comparison_markdown,
    load_baseline,
    record_run_baseline,
    save_baseline,
)
from agenttester.git_manager import DiffStats
from agenttester.metrics import TokenUsage


def _mock_git() -> MagicMock:
    git = MagicMock()
    git.get_diff_stats.return_value = DiffStats(
        files_changed=1, insertions=5, deletions=0, changed_files=["a.py"]
    )
    return git


class TestBaselineIO:
    def test_round_trip(self, tmp_path: Path) -> None:
        baseline = BaselineFile(suite="demo")
        record_run_baseline(
            baseline,
            "run-1",
            case_id="c1",
            prompt="hello",
            results=[AgentResult("claude", 0, 5.0, "", "", None)],
            git=_mock_git(),
            base_ref="abc123",
        )
        path = tmp_path / "baseline.json"
        save_baseline(path, baseline)
        loaded = load_baseline(path)
        assert loaded.suite == "demo"
        assert "run-1" in loaded.runs
        assert loaded.runs["run-1"].agents["claude"].duration == 5.0


class TestCompareBaseline:
    def test_detects_duration_regression(self) -> None:
        baseline = BaselineFile()
        record_run_baseline(
            baseline,
            "run-1",
            case_id=None,
            prompt="p",
            results=[AgentResult("claude", 0, 10.0, "", "", None)],
            git=_mock_git(),
            base_ref="abc",
        )
        current = [AgentResult("claude", 0, 20.0, "", "", None)]
        comps = compare_run_to_baseline("run-1", current, _mock_git(), "abc", baseline)
        assert comps[0].has_regression is True

    def test_detects_exit_regression(self) -> None:
        baseline = BaselineFile()
        record_run_baseline(
            baseline,
            "run-1",
            case_id=None,
            prompt="p",
            results=[AgentResult("claude", 0, 5.0, "", "", None)],
            git=_mock_git(),
            base_ref="abc",
        )
        current = [AgentResult("claude", 1, 5.0, "", "", "failed")]
        comps = compare_run_to_baseline("run-1", current, _mock_git(), "abc", baseline)
        assert comps[0].has_regression is True

    def test_markdown_output(self) -> None:
        baseline = BaselineFile()
        record_run_baseline(
            baseline,
            "run-1",
            case_id=None,
            prompt="p",
            results=[
                AgentResult(
                    "cursor",
                    0,
                    5.0,
                    "",
                    "",
                    None,
                    usage=TokenUsage(input=10, output=5),
                )
            ],
            git=_mock_git(),
            base_ref="abc",
        )
        current = [
            AgentResult(
                "cursor",
                0,
                5.0,
                "",
                "",
                None,
                usage=TokenUsage(input=50, output=5),
            )
        ]
        comps = compare_run_to_baseline("run-1", current, _mock_git(), "abc", baseline)
        md = format_comparison_markdown(comps)
        assert any("regression" in line for line in md)
