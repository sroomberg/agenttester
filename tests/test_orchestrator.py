"""Tests for agenttester.orchestrator."""

from __future__ import annotations

from agenttester.orchestrator import _inject_branch_into_prompt


class TestInjectBranchIntoPrompt:
    def test_format(self) -> None:
        prompt = "Fix the bug"
        result = _inject_branch_into_prompt(prompt, "abc123", "claude")
        assert result.startswith("You are working on branch `agenttester/abc123/claude`.\n\n")
        assert result.endswith(prompt)

    def test_includes_run_id_and_agent_name(self) -> None:
        result = _inject_branch_into_prompt("test", "run123", "myagent")
        assert "run123" in result
        assert "myagent" in result
        assert "agenttester/run123/myagent" in result

    def test_preserves_original_prompt(self) -> None:
        original = "This is a multi-line\nprompt with special chars: !@#$%"
        result = _inject_branch_into_prompt(original, "x", "y")
        assert original in result

    def test_branch_is_formatted_as_code(self) -> None:
        result = _inject_branch_into_prompt("", "run1", "agent1")
        assert "`agenttester/run1/agent1`" in result
