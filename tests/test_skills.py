"""Tests for agenttester.skills."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agenttester.skills import load_skills


class TestLoadSkills:
    def test_builtin_skills_always_loaded(self) -> None:
        result = load_skills()
        assert len(result) > 0

    def test_builtin_editing_skill_present(self) -> None:
        result = load_skills()
        assert "edit" in result.lower()

    def test_builtin_testing_skill_present(self) -> None:
        result = load_skills()
        assert "test" in result.lower()

    def test_builtin_git_skill_present(self) -> None:
        result = load_skills()
        assert "git" in result.lower()

    def test_builtin_git_skill_forbids_default_branch_push(self) -> None:
        result = load_skills()
        assert "default branch" in result.lower()

    def test_builtin_bash_skill_present(self) -> None:
        result = load_skills()
        assert "bash" in result.lower()

    def test_builtin_bash_skill_scoped_to_repo(self) -> None:
        result = load_skills()
        assert "worktree" in result.lower()

    def test_local_skill_overrides_builtin(self, tmp_path: Path) -> None:
        local_skills = tmp_path / ".agent-tester" / "skills"
        local_skills.mkdir(parents=True)
        (local_skills / "editing.md").write_text("Custom editing rule.")
        result = load_skills(tmp_path)
        assert "Custom editing rule." in result

    def test_local_skill_supplements_builtins(self, tmp_path: Path) -> None:
        local_skills = tmp_path / ".agent-tester" / "skills"
        local_skills.mkdir(parents=True)
        (local_skills / "extra.md").write_text("Always write docstrings.")
        result = load_skills(tmp_path)
        assert "Always write docstrings." in result
        assert "test" in result.lower()

    def test_global_skill_overrides_builtin(self, tmp_path: Path) -> None:
        global_skills = tmp_path / "global_skills"
        global_skills.mkdir()
        (global_skills / "testing.md").write_text("Global testing rule.")
        target = "agenttester.skills._get_global_skills_dir"
        with patch(target, return_value=global_skills):
            result = load_skills()
            assert "Global testing rule." in result

    def test_local_overrides_global(self, tmp_path: Path) -> None:
        global_skills = tmp_path / "global_skills"
        global_skills.mkdir()
        (global_skills / "editing.md").write_text("Global editing rule.")
        local_skills = tmp_path / "repo" / ".agent-tester" / "skills"
        local_skills.mkdir(parents=True)
        (local_skills / "editing.md").write_text("Local editing rule.")
        target = "agenttester.skills._get_global_skills_dir"
        with patch(target, return_value=global_skills):
            result = load_skills(tmp_path / "repo")
            assert "Local editing rule." in result
            assert "Global editing rule." not in result

    def test_no_local_dir_falls_back_to_builtins(self, tmp_path: Path) -> None:
        result = load_skills(tmp_path)
        assert "test" in result.lower()

    def test_local_skill_appears_after_builtins(self, tmp_path: Path) -> None:
        local_skills = tmp_path / ".agent-tester" / "skills"
        local_skills.mkdir(parents=True)
        (local_skills / "zzz-custom.md").write_text("My custom rule.")
        result = load_skills(tmp_path)
        assert result.index("My custom rule.") > result.index("edit")

    def test_local_override_appears_after_unoverridden_builtins(
        self, tmp_path: Path
    ) -> None:
        local_skills = tmp_path / ".agent-tester" / "skills"
        local_skills.mkdir(parents=True)
        (local_skills / "editing.md").write_text("Custom editing rule.")
        result = load_skills(tmp_path)
        # local override should appear after the other built-ins (e.g. testing)
        assert "Custom editing rule." in result
        assert result.index("Custom editing rule.") > result.index("test")

    def test_global_skill_appears_after_builtins(self, tmp_path: Path) -> None:
        global_skills = tmp_path / "global_skills"
        global_skills.mkdir()
        (global_skills / "zzz-global.md").write_text("Global custom rule.")
        target = "agenttester.skills._get_global_skills_dir"
        with patch(target, return_value=global_skills):
            result = load_skills()
        assert result.index("Global custom rule.") > result.index("edit")

    def test_local_skill_appears_after_global(self, tmp_path: Path) -> None:
        global_skills = tmp_path / "global_skills"
        global_skills.mkdir()
        (global_skills / "zzz-global.md").write_text("Global rule.")
        local_skills = tmp_path / "repo" / ".agent-tester" / "skills"
        local_skills.mkdir(parents=True)
        (local_skills / "zzz-local.md").write_text("Local rule.")
        target = "agenttester.skills._get_global_skills_dir"
        with patch(target, return_value=global_skills):
            result = load_skills(tmp_path / "repo")
        assert result.index("Local rule.") > result.index("Global rule.")
