"""Tests for agenttester.orchestrator."""

from __future__ import annotations

from agenttester.orchestrator import _build_prompt


class TestBuildPrompt:
    def test_includes_branch(self) -> None:
        result = _build_prompt("Fix the bug", "abc123", "claude", "")
        assert "`agenttester/abc123/claude`" in result

    def test_includes_run_id_and_agent_name(self) -> None:
        result = _build_prompt("test", "run123", "myagent", "")
        assert "agenttester/run123/myagent" in result

    def test_preserves_original_prompt(self) -> None:
        original = "This is a multi-line\nprompt with special chars: !@#$%"
        result = _build_prompt(original, "x", "y", "")
        assert original in result

    def test_skills_prepended_before_branch(self) -> None:
        result = _build_prompt("do the thing", "r1", "claude", "Always edit freely.")
        skills_pos = result.index("Always edit freely.")
        branch_pos = result.index("agenttester/r1/claude")
        prompt_pos = result.index("do the thing")
        assert skills_pos < branch_pos < prompt_pos

    def test_no_skills_still_includes_branch_and_prompt(self) -> None:
        result = _build_prompt("my task", "r1", "agent1", "")
        assert "agenttester/r1/agent1" in result
        assert "my task" in result

    def test_empty_skills_not_double_newlined(self) -> None:
        result = _build_prompt("task", "r1", "a1", "")
        assert not result.startswith("\n")
