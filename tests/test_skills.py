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
