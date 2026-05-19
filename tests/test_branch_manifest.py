"""Tests for agenttester.branch_manifest."""

from __future__ import annotations

from pathlib import Path

from agenttester.branch_manifest import list_branches, record_branch, remove_branch


class TestRecordBranch:
    def test_creates_entry(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        record_branch("agenttester/m/abc-feat", "git@gh:user/repo.git", "sess1", p)
        data = list_branches(p)
        assert "agenttester/m/abc-feat" in data
        assert data["agenttester/m/abc-feat"]["remote"] == "git@gh:user/repo.git"
        assert data["agenttester/m/abc-feat"]["session_id"] == "sess1"
        assert "created_at" in data["agenttester/m/abc-feat"]

    def test_multiple_branches_coexist(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        record_branch("agenttester/m/abc-feat", "git@gh:user/repo.git", "s1", p)
        record_branch("agenttester/n/abc-feat", "git@gh:user/repo.git", "s1", p)
        data = list_branches(p)
        assert len(data) == 2

    def test_overwrites_existing_entry(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        record_branch("agenttester/m/abc-feat", "git@gh:user/repo.git", "s1", p)
        record_branch("agenttester/m/abc-feat", "git@gh:user/other.git", "s2", p)
        data = list_branches(p)
        assert len(data) == 1
        assert data["agenttester/m/abc-feat"]["remote"] == "git@gh:user/other.git"
        assert data["agenttester/m/abc-feat"]["session_id"] == "s2"

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        p = tmp_path / "nested" / "dir" / "branches.json"
        record_branch("agenttester/m/abc-feat", "git@gh:user/repo.git", "s1", p)
        assert p.exists()


class TestRemoveBranch:
    def test_removes_existing_entry(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        record_branch("agenttester/m/abc-feat", "git@gh:user/repo.git", "s1", p)
        remove_branch("agenttester/m/abc-feat", p)
        assert list_branches(p) == {}

    def test_leaves_other_entries_intact(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        record_branch("agenttester/m/abc-feat", "git@gh:user/repo.git", "s1", p)
        record_branch("agenttester/n/abc-feat", "git@gh:user/repo.git", "s1", p)
        remove_branch("agenttester/m/abc-feat", p)
        data = list_branches(p)
        assert "agenttester/m/abc-feat" not in data
        assert "agenttester/n/abc-feat" in data

    def test_no_error_when_branch_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        remove_branch("agenttester/m/nonexistent", p)  # should not raise

    def test_no_error_when_file_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        remove_branch("agenttester/m/abc-feat", p)  # file doesn't exist yet


class TestListBranches:
    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        assert list_branches(p) == {}

    def test_returns_all_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "branches.json"
        record_branch("agenttester/a/slug", "remote-a", "s1", p)
        record_branch("agenttester/b/slug", "remote-b", "s2", p)
        data = list_branches(p)
        assert set(data.keys()) == {"agenttester/a/slug", "agenttester/b/slug"}
